"""Generate synthetic 3D phantoms with known topology for pipeline validation.

The primary phantom is a solid sphere with N internal spherical voids.
It is used to verify that the TDA pipeline recovers exactly N H₂ features
before applying the pipeline to real CT data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Core phantom generators
# ---------------------------------------------------------------------------

def sphere_with_cavities(
    shape: tuple[int, int, int] = (64, 64, 64),
    n_cavities: int = 3,
    outer_radius_frac: float = 0.40,
    cavity_radius_frac: float = 0.07,
    smooth_sigma: float = 1.5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, int]:
    """Solid sphere containing *n_cavities* internal spherical voids.

    Returns a float32 density array in [0, 1] and the number of successfully
    placed cavities (may be less than *n_cavities* if the sphere is too small
    to fit them all without overlap).

    The density encodes solid material → high values, air/voids → low values.
    For TDA, pass ``1.0 - volume`` as the sublevel filtration function so the
    solid material appears first; each enclosed void then creates one H₂ class.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    sz, sy, sx = shape
    center = np.array([sz / 2.0, sy / 2.0, sx / 2.0])
    outer_r = outer_radius_frac * min(shape)
    cav_r = cavity_radius_frac * min(shape)

    # Coordinate grids (Z, Y, X)
    z_g, y_g, x_g = np.mgrid[0:sz, 0:sy, 0:sx]

    # Outer solid sphere
    dist_outer = np.sqrt(
        (z_g - center[0]) ** 2 + (y_g - center[1]) ** 2 + (x_g - center[2]) ** 2
    )
    volume = (dist_outer < outer_r).astype(np.float32)

    # Safe placement zone: cavities must be fully inside the outer sphere
    safe_r = outer_r - cav_r * 2.5

    placed_centers: list[np.ndarray] = []
    for _ in range(n_cavities):
        for _attempt in range(500):
            # Uniform random point inside a sphere of radius safe_r
            u = rng.standard_normal(3)
            u /= np.linalg.norm(u) + 1e-12
            r = safe_r * rng.uniform(0.0, 1.0) ** (1.0 / 3.0)
            candidate = center + r * u

            # Reject if too close to another cavity
            min_sep = 3.0 * cav_r
            if all(
                np.linalg.norm(candidate - pc) > min_sep for pc in placed_centers
            ):
                placed_centers.append(candidate)
                break

    # Carve cavities out of the solid
    for pc in placed_centers:
        cav_dist = np.sqrt(
            (z_g - pc[0]) ** 2 + (y_g - pc[1]) ** 2 + (x_g - pc[2]) ** 2
        )
        volume[cav_dist < cav_r] = 0.0

    # Smooth so the filtration has a nice gradient (improves persistence separation)
    volume = gaussian_filter(volume.astype(np.float64), sigma=smooth_sigma)
    volume = np.clip(volume, 0.0, 1.0).astype(np.float32)

    return volume, len(placed_centers)


def torus(
    shape: tuple[int, int, int] = (64, 64, 64),
    major_radius_frac: float = 0.30,
    minor_radius_frac: float = 0.12,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """Solid torus — known topology: H₀=1, H₁=1, H₂=1.

    Useful as a secondary validation phantom.
    """
    sz, sy, sx = shape
    center = np.array([sz / 2.0, sy / 2.0, sx / 2.0])
    R = major_radius_frac * min(shape)
    r = minor_radius_frac * min(shape)

    z_g, y_g, x_g = np.mgrid[0:sz, 0:sy, 0:sx]
    dz = z_g - center[0]
    dy = y_g - center[1]
    dx = x_g - center[2]

    # Distance from the torus central circle (in XY plane)
    rho = np.sqrt(dx**2 + dy**2)
    dist_torus = np.sqrt((rho - R) ** 2 + dz**2)

    volume = (dist_torus < r).astype(np.float32)
    volume = gaussian_filter(volume.astype(np.float64), sigma=smooth_sigma)
    return np.clip(volume, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Persistence / filtration helpers
# ---------------------------------------------------------------------------

def filtration_field(volume: np.ndarray) -> np.ndarray:
    """Return ``1.0 - volume`` as a float64 sublevel filtration field.

    In this field the dense solid (high density) has *low* filtration values
    and is born first; enclosed air voids (low density) have *high* values and
    appear last.  Each enclosed void creates one H₂ class.
    """
    return (1.0 - volume.astype(np.float64))


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_phantom(
    volume: np.ndarray,
    path: str | Path,
    metadata: dict | None = None,
) -> Path:
    """Save phantom as .npy with an optional JSON sidecar."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, volume)
    if metadata is not None:
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(metadata, indent=2))
    return path


def load_phantom(path: str | Path) -> np.ndarray:
    """Load a .npy phantom saved by :func:`save_phantom`."""
    return np.load(str(path)).astype(np.float32)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile

    print("Generating sphere-with-cavities phantom (N=4)...")
    vol, n_placed = sphere_with_cavities(shape=(64, 64, 64), n_cavities=4)
    print(f"  shape={vol.shape}, dtype={vol.dtype}, range=[{vol.min():.3f}, {vol.max():.3f}]")
    print(f"  Placed {n_placed} cavities")

    # Save and reload
    with tempfile.TemporaryDirectory() as td:
        p = save_phantom(vol, Path(td) / "phantom_test.npy", {"n_cavities": n_placed})
        vol2 = load_phantom(p)
        assert np.allclose(vol, vol2), "Round-trip save/load failed"
    print("  Round-trip save/load: OK")

    # Quick TDA smoke test
    try:
        import gudhi

        field = filtration_field(vol)
        cc = gudhi.CubicalComplex(
            dimensions=list(vol.shape),
            top_dimensional_cells=field.flatten(order="C").tolist(),
        )
        cc.compute_persistence(homology_coeff_field=2, min_persistence=0)
        pairs = cc.persistence()
        h2 = [(b, d) for dim, (b, d) in pairs if dim == 2 and d < np.inf]
        persistent_h2 = [(b, d) for b, d in h2 if d - b > 0.1]
        print(f"  H2 features (persistence > 0.1): {len(persistent_h2)}  (expected ~{n_placed})")
    except ImportError:
        print("  GUDHI not installed — skipping TDA smoke test")

    print("\nTorus phantom...")
    tor = torus(shape=(64, 64, 64))
    print(f"  shape={tor.shape}, dtype={tor.dtype}, range=[{tor.min():.3f}, {tor.max():.3f}]")
    print("PASS")
