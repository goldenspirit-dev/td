"""Convert persistence diagrams to fixed-size feature vectors.

Three representations are provided:

Persistence image (PI)
    A 2-D histogram of the persistence diagram weighted by persistence,
    smoothed with a Gaussian kernel.  Flat vector suitable for ML classifiers.

Betti curve (BC)
    The rank of the homology group as a function of the filtration threshold.
    Compact (1-D), easy to interpret.

Persistence landscape (PL)
    A stack of piecewise-linear functions enveloping the diagram.
    Provides stable L² statistics.

All representations use ``gudhi.representations`` where available and fall
back to pure-NumPy implementations otherwise.
"""
from __future__ import annotations

import numpy as np

try:
    from gudhi.representations import PersistenceImage as _GPI
    from gudhi.representations import BettiCurve as _GBC
    from gudhi.representations import Landscape as _GLS
    _GUDHI_REPR = True
except (ImportError, ValueError):
    # ValueError can occur with numpy ABI mismatches (numpy 2.x + sklearn 1.3.x).
    # Pure-NumPy fallbacks are used in that case.
    _GUDHI_REPR = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finite(diagram: np.ndarray) -> np.ndarray:
    """Return rows of *diagram* where death is finite."""
    if len(diagram) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return diagram[~np.isinf(diagram[:, 1])].astype(np.float64)


def _auto_range(diagram: np.ndarray) -> tuple[float, float]:
    d = _finite(diagram)
    if len(d) == 0:
        return 0.0, 1.0
    return float(d.min()), float(d.max())


# ---------------------------------------------------------------------------
# Persistence image
# ---------------------------------------------------------------------------

def persistence_image(
    diagram: np.ndarray,
    *,
    resolution: tuple[int, int] = (20, 20),
    bandwidth: float = 0.05,
    weight_power: float = 1.0,
    im_range: list[float] | None = None,
) -> np.ndarray:
    """Compute a persistence image for a single diagram.

    Parameters
    ----------
    diagram:
        Shape (N, 2) array of [birth, death] pairs.  Infinite bars are ignored.
    resolution:
        Grid resolution (rows × cols) of the output image.
    bandwidth:
        Gaussian kernel σ.  Smaller values give sharper images.
    weight_power:
        Points are weighted by (death - birth) ^ weight_power.
    im_range:
        [x_min, x_max, y_min, y_max] for the birth–persistence rectangle.
        Inferred from the diagram automatically if None.

    Returns
    -------
    np.ndarray
        Flat float64 array of length ``resolution[0] * resolution[1]``.
    """
    d = _finite(diagram)

    if im_range is None:
        if len(d) == 0:
            im_range = [0.0, 1.0, 0.0, 1.0]
        else:
            b_min, b_max = d[:, 0].min(), d[:, 0].max()
            p_max = (d[:, 1] - d[:, 0]).max()
            margin = max((b_max - b_min) * 0.1, p_max * 0.1, 0.05)
            im_range = [b_min - margin, b_max + margin, 0.0, p_max + margin]

    if _GUDHI_REPR:
        pi = _GPI(
            bandwidth=bandwidth,
            weight=lambda x: x[1] ** weight_power,
            resolution=list(resolution),
            im_range=im_range,
        )
        return pi.fit_transform([d]).flatten()

    return _persistence_image_numpy(d, resolution, bandwidth, weight_power, im_range)


def _persistence_image_numpy(
    diagram: np.ndarray,
    resolution: tuple[int, int],
    bandwidth: float,
    weight_power: float,
    im_range: list[float],
) -> np.ndarray:
    rows, cols = resolution
    x_min, x_max, y_min, y_max = im_range
    xs = np.linspace(x_min, x_max, cols)
    ys = np.linspace(y_min, y_max, rows)
    image = np.zeros((rows, cols), dtype=np.float64)

    if len(diagram) == 0:
        return image.flatten()

    births = diagram[:, 0]
    persistences = diagram[:, 1] - diagram[:, 0]
    weights = persistences ** weight_power
    for b, p, w in zip(births, persistences, weights):
        gx = np.exp(-0.5 * ((xs - b) / bandwidth) ** 2)
        gy = np.exp(-0.5 * ((ys - p) / bandwidth) ** 2)
        image += w * np.outer(gy, gx)
    return image.flatten()


# ---------------------------------------------------------------------------
# Betti curve
# ---------------------------------------------------------------------------

def betti_curve(
    diagram: np.ndarray,
    *,
    resolution: int = 100,
    sample_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Betti curve for a single diagram.

    The Betti number β(t) = number of bars [birth, death) with birth ≤ t < death.

    Returns
    -------
    thresholds : np.ndarray  shape (resolution,)
    betti_values : np.ndarray  shape (resolution,)  float64
    """
    d = _finite(diagram)
    if sample_range is None:
        sample_range = _auto_range(d) if len(d) > 0 else (0.0, 1.0)

    thresholds = np.linspace(sample_range[0], sample_range[1], resolution)

    if _GUDHI_REPR:
        bc = _GBC(resolution=resolution, sample_range=list(sample_range))
        values = bc.fit_transform([d]).flatten()
        return thresholds, values.astype(np.float64)

    values = np.zeros(resolution, dtype=np.float64)
    if len(d) > 0:
        for i, t in enumerate(thresholds):
            values[i] = np.sum((d[:, 0] <= t) & (d[:, 1] > t))
    return thresholds, values


# ---------------------------------------------------------------------------
# Persistence landscape
# ---------------------------------------------------------------------------

def persistence_landscape(
    diagram: np.ndarray,
    *,
    num_landscapes: int = 5,
    resolution: int = 100,
    sample_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Compute persistence landscapes.  Returns flat array of length num_landscapes*resolution."""
    d = _finite(diagram)
    if sample_range is None:
        sample_range = _auto_range(d) if len(d) > 0 else (0.0, 1.0)

    if _GUDHI_REPR:
        ls = _GLS(
            num_landscapes=num_landscapes,
            resolution=resolution,
            sample_range=list(sample_range),
        )
        return ls.fit_transform([d]).flatten()

    return _landscape_numpy(d, num_landscapes, resolution, sample_range)


def _landscape_numpy(
    diagram: np.ndarray,
    num_landscapes: int,
    resolution: int,
    sample_range: tuple[float, float],
) -> np.ndarray:
    thresholds = np.linspace(sample_range[0], sample_range[1], resolution)
    landscapes = np.zeros((num_landscapes, resolution), dtype=np.float64)
    if len(diagram) == 0:
        return landscapes.flatten()
    for i, t in enumerate(thresholds):
        tent_values = np.maximum(
            0.0, np.minimum(t - diagram[:, 0], diagram[:, 1] - t)
        )
        sorted_vals = np.sort(tent_values)[::-1]
        for k in range(min(num_landscapes, len(sorted_vals))):
            landscapes[k, i] = sorted_vals[k]
    return landscapes.flatten()


# ---------------------------------------------------------------------------
# Convenience: vectorise a full diagram dict
# ---------------------------------------------------------------------------

def vectorize_diagrams(
    diagrams: dict[int, np.ndarray],
    dims: tuple[int, ...] = (0, 1, 2),
    *,
    pi_resolution: tuple[int, int] = (20, 20),
    bc_resolution: int = 100,
    bandwidth: float = 0.05,
) -> dict[str, np.ndarray]:
    """Compute persistence images and Betti curves for multiple H-dimensions.

    Returns dict with keys ``"pi_0"``, ``"bc_0"``, ``"pi_1"``, ``"bc_1"``, …
    """
    result: dict[str, np.ndarray] = {}
    for dim in dims:
        diag = diagrams.get(dim, np.empty((0, 2)))
        result[f"pi_{dim}"] = persistence_image(
            diag, resolution=pi_resolution, bandwidth=bandwidth
        )
        _, bc = betti_curve(diag, resolution=bc_resolution)
        result[f"bc_{dim}"] = bc
    return result


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from tda_fruit_ct.synthetic import sphere_with_cavities, filtration_field
    from tda_fruit_ct.filtration import sublevel_filtration
    from tda_fruit_ct.persistence import compute_persistence

    print(f"Using gudhi.representations: {_GUDHI_REPR}")
    print("Testing on 32x32x32 phantom with 3 cavities...\n")
    vol, n = sphere_with_cavities(shape=(32, 32, 32), n_cavities=3)
    diagrams = compute_persistence(sublevel_filtration(filtration_field(vol)))

    pi = persistence_image(diagrams[2], resolution=(10, 10))
    print(f"Persistence image (H2): shape={pi.shape}, sum={pi.sum():.4f}, max={pi.max():.4f}")

    thresholds, bc = betti_curve(diagrams[2], resolution=50)
    print(f"Betti curve (H2): max betti2 = {bc.max():.0f}")

    ls = persistence_landscape(diagrams[2], num_landscapes=3, resolution=50)
    print(f"Landscape (H2): shape={ls.shape}, max={ls.max():.4f}")

    vecs = vectorize_diagrams(diagrams)
    print("\nvectorize_diagrams:")
    for k, v in sorted(vecs.items()):
        print(f"  {k}: {v.shape}")

    print("\nPASS")
