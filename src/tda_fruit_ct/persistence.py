"""Compute persistent homology diagrams from GUDHI cubical complexes.

Typical usage
-------------
>>> from tda_fruit_ct.filtration import sublevel_filtration
>>> from tda_fruit_ct.synthetic import sphere_with_cavities, filtration_field
>>> vol, n_cav = sphere_with_cavities()
>>> cc = sublevel_filtration(filtration_field(vol))
>>> diagrams = compute_persistence(cc)          # {0: array, 1: array, 2: array}
>>> h2 = diagrams[2]                            # shape (N, 2), columns (birth, death)
>>> sig = filter_noise(h2, min_persistence=0.1) # remove near-diagonal noise
>>> print(f"Significant H₂ features: {len(sig)}")
"""
from __future__ import annotations

import numpy as np

try:
    import gudhi
except ImportError as _exc:
    raise ImportError("GUDHI is required: pip install gudhi") from _exc


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_persistence(
    cc: "gudhi.CubicalComplex",
    *,
    coeff_field: int = 2,
    min_persistence: float = 0.0,
) -> dict[int, np.ndarray]:
    """Compute persistent homology and return diagrams keyed by dimension.

    Parameters
    ----------
    cc:
        A GUDHI CubicalComplex (from :func:`tda_fruit_ct.filtration.build_filtration`).
        The complex must *not* have been fitted yet (this function calls
        ``compute_persistence`` internally).
    coeff_field:
        Prime coefficient field for homology coefficients (2 = Z/2Z, default).
    min_persistence:
        Pairs with ``death - birth < min_persistence`` are omitted.
        Use 0 to keep all pairs including noise.

    Returns
    -------
    dict[int, np.ndarray]
        Keys are homology dimensions (0, 1, 2, …).
        Values are float64 arrays of shape (N, 2) with columns ``[birth, death]``.
        Infinite bars (essential classes) have ``death = np.inf``.
    """
    cc.compute_persistence(
        homology_coeff_field=coeff_field,
        min_persistence=min_persistence,
    )
    raw = cc.persistence()  # list of (dim, (birth, death))

    diagrams: dict[int, list[tuple[float, float]]] = {}
    for dim, (birth, death) in raw:
        diagrams.setdefault(dim, []).append((float(birth), float(death)))

    result: dict[int, np.ndarray] = {}
    for dim, pairs in diagrams.items():
        arr = np.array(pairs, dtype=np.float64)
        result[dim] = arr

    # Ensure all requested dims are present, even if empty
    for dim in (0, 1, 2):
        if dim not in result:
            result[dim] = np.empty((0, 2), dtype=np.float64)

    return result


# ---------------------------------------------------------------------------
# Diagram manipulation
# ---------------------------------------------------------------------------

def filter_noise(
    diagram: np.ndarray,
    min_persistence: float = 0.1,
    *,
    keep_infinite: bool = True,
) -> np.ndarray:
    """Remove near-diagonal (low-persistence) features from a diagram.

    Parameters
    ----------
    diagram:
        Shape (N, 2) array of [birth, death] pairs (from :func:`compute_persistence`).
    min_persistence:
        Minimum ``death - birth`` to keep a feature.
    keep_infinite:
        If True, infinite bars (``death = inf``) are always kept.

    Returns
    -------
    np.ndarray
        Filtered diagram, same column layout as input.
    """
    if len(diagram) == 0:
        return diagram

    persistence = diagram[:, 1] - diagram[:, 0]
    is_infinite = np.isinf(diagram[:, 1])

    if keep_infinite:
        mask = (persistence >= min_persistence) | is_infinite
    else:
        mask = persistence >= min_persistence

    return diagram[mask]


def finite_diagram(diagram: np.ndarray) -> np.ndarray:
    """Return only the finite bars (drop ``death = inf`` rows)."""
    if len(diagram) == 0:
        return diagram
    return diagram[~np.isinf(diagram[:, 1])]


def persistence_values(diagram: np.ndarray) -> np.ndarray:
    """Return ``death - birth`` for all finite pairs, sorted descending."""
    d = finite_diagram(diagram)
    if len(d) == 0:
        return np.array([], dtype=np.float64)
    return np.sort(d[:, 1] - d[:, 0])[::-1]


def count_significant(
    diagram: np.ndarray,
    min_persistence: float = 0.1,
) -> int:
    """Count features with persistence ≥ *min_persistence*."""
    return int(len(filter_noise(diagram, min_persistence, keep_infinite=True)))


def summary(diagrams: dict[int, np.ndarray], min_persistence: float = 0.05) -> str:
    """Return a compact text summary of persistence diagrams.

    Example output::

        H₀: 1 total, 1 significant (p≥0.05)
        H₁: 0 total, 0 significant (p≥0.05)
        H₂: 5 total, 3 significant (p≥0.05)
    """
    lines = []
    for dim in sorted(diagrams):
        diag = diagrams[dim]
        total = len(diag)
        sig = count_significant(diag, min_persistence)
        lines.append(f"H{dim}: {total} total, {sig} significant (p>={min_persistence})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from tda_fruit_ct.synthetic import sphere_with_cavities, filtration_field
    from tda_fruit_ct.filtration import sublevel_filtration

    N = 3
    print(f"Phantom with {N} cavities (32x32x32)...")
    vol, n_placed = sphere_with_cavities(shape=(32, 32, 32), n_cavities=N)
    print(f"  Placed {n_placed} cavities")

    cc = sublevel_filtration(filtration_field(vol))
    diagrams = compute_persistence(cc, min_persistence=0.0)

    print("\nRaw persistence summary:")
    print(summary(diagrams, min_persistence=0.0))

    print("\nFiltered (p >= 0.1):")
    for dim in (0, 1, 2):
        sig = count_significant(diagrams[dim], 0.1)
        print(f"  H{dim}: {sig} features")

    h2_sig = count_significant(diagrams[2], 0.1)
    verdict = "PASS" if h2_sig == n_placed else f"WARN: got {h2_sig}, expected {n_placed}"
    print(f"\nH2 validation: {verdict}")
