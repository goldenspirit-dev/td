"""Plotting utilities for CT volumes and TDA outputs.

All functions return a ``matplotlib.figure.Figure`` and optionally save it.
Pass ``save_path=None`` (default) to skip saving.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Colour palette for homology dimensions
_DIM_COLOURS = {0: "#2196F3", 1: "#FF9800", 2: "#4CAF50"}


# ---------------------------------------------------------------------------
# Volume visualisation
# ---------------------------------------------------------------------------

def plot_slices(
    volume: np.ndarray,
    title: str = "CT volume",
    *,
    cmap: str = "gray",
    save_path: Optional[str | Path] = None,
    figsize: tuple[float, float] = (12, 4),
) -> plt.Figure:
    """Show axial, coronal, and sagittal mid-slices of a 3-D volume.

    Parameters
    ----------
    volume:
        float32 array, shape (Z, Y, X), values in [0, 1].
    """
    sz, sy, sx = volume.shape
    slices = [
        (volume[sz // 2, :, :], "Axial (Z mid)"),
        (volume[:, sy // 2, :], "Coronal (Y mid)"),
        (volume[:, :, sx // 2], "Sagittal (X mid)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, (slc, label) in zip(axes, slices):
        im = ax.imshow(slc, cmap=cmap, origin="lower", vmin=0, vmax=1)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_intensity_histogram(
    volume: np.ndarray,
    title: str = "Intensity histogram",
    *,
    bins: int = 100,
    save_path: Optional[str | Path] = None,
    figsize: tuple[float, float] = (8, 4),
) -> plt.Figure:
    """Plot a histogram of voxel intensities."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(volume.flatten(), bins=bins, color="#607D8B", edgecolor="none", alpha=0.8)
    ax.set_xlabel("Normalised intensity")
    ax.set_ylabel("Voxel count")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Persistence diagram
# ---------------------------------------------------------------------------

def plot_persistence_diagram(
    diagrams: dict[int, np.ndarray],
    title: str = "Persistence diagram",
    *,
    min_persistence: float = 0.0,
    max_death: Optional[float] = None,
    save_path: Optional[str | Path] = None,
    figsize: tuple[float, float] = (6, 6),
) -> plt.Figure:
    """Scatter-plot of (birth, death) pairs for each homology dimension.

    Dimensions 0, 1, 2 are plotted in blue, orange, green respectively.
    Near-diagonal features (persistence < min_persistence) are shown as
    small, semi-transparent markers.
    """
    fig, ax = plt.subplots(figsize=figsize)

    all_finite: list[float] = []
    for dim, diag in sorted(diagrams.items()):
        if len(diag) == 0:
            continue
        colour = _DIM_COLOURS.get(dim, "gray")
        finite = diag[~np.isinf(diag[:, 1])]
        infinite = diag[np.isinf(diag[:, 1])]
        all_finite.extend(finite.flatten().tolist())

        persistence = finite[:, 1] - finite[:, 0]
        sig = persistence >= min_persistence
        noise = ~sig

        ax.scatter(finite[sig, 0], finite[sig, 1],
                   c=colour, s=30, alpha=0.9, label=f"H{dim}", zorder=3)
        if noise.any():
            ax.scatter(finite[noise, 0], finite[noise, 1],
                       c=colour, s=8, alpha=0.25, zorder=2)
        # Draw infinite bars as triangles at the top axis edge
        if len(infinite) > 0 and max_death is not None:
            ax.scatter(infinite[:, 0], np.full(len(infinite), max_death * 0.97),
                       marker="^", c=colour, s=40, alpha=0.8, zorder=3)

    lim = max(all_finite, default=1.0) * 1.05
    if max_death is not None:
        lim = max(lim, max_death * 1.05)
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.4, zorder=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Betti curves
# ---------------------------------------------------------------------------

def plot_betti_curves(
    betti_data: dict[int, tuple[np.ndarray, np.ndarray]],
    title: str = "Betti curves",
    *,
    save_path: Optional[str | Path] = None,
    figsize: tuple[float, float] = (8, 4),
) -> plt.Figure:
    """Plot Betti curves for multiple homology dimensions.

    Parameters
    ----------
    betti_data:
        Dict mapping dimension → (thresholds, values).  The values come from
        :func:`tda_fruit_ct.vectorization.betti_curve`.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for dim, (thresholds, values) in sorted(betti_data.items()):
        colour = _DIM_COLOURS.get(dim, "gray")
        ax.plot(thresholds, values, color=colour, lw=2, label=f"β{dim}")

    ax.set_xlabel("Filtration threshold")
    ax.set_ylabel("Betti number")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Persistence image
# ---------------------------------------------------------------------------

def plot_persistence_image(
    pi_flat: np.ndarray,
    resolution: tuple[int, int] = (20, 20),
    title: str = "Persistence image",
    *,
    cmap: str = "viridis",
    save_path: Optional[str | Path] = None,
    figsize: tuple[float, float] = (5, 4.5),
) -> plt.Figure:
    """Display a persistence image as a heat map.

    Parameters
    ----------
    pi_flat:
        Flat array of length resolution[0] * resolution[1], from
        :func:`tda_fruit_ct.vectorization.persistence_image`.
    """
    image = pi_flat.reshape(resolution)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        image, origin="lower", cmap=cmap, aspect="equal",
        norm=mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=image.max() + 1e-12),
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Birth (normalised)")
    ax.set_ylabel("Persistence (normalised)")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------

def plot_diagram_comparison(
    diagrams_a: dict[int, np.ndarray],
    diagrams_b: dict[int, np.ndarray],
    labels: tuple[str, str] = ("Phantom", "Real CT"),
    title: str = "Persistence diagram comparison",
    *,
    dims: tuple[int, ...] = (0, 1, 2),
    save_path: Optional[str | Path] = None,
    figsize: tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Side-by-side persistence diagrams for two samples."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for ax, diags, label in zip(axes, (diagrams_a, diagrams_b), labels):
        all_finite: list[float] = []
        for dim in dims:
            diag = diags.get(dim, np.empty((0, 2)))
            if len(diag) == 0:
                continue
            colour = _DIM_COLOURS.get(dim, "gray")
            finite = diag[~np.isinf(diag[:, 1])]
            all_finite.extend(finite.flatten().tolist())
            ax.scatter(finite[:, 0], finite[:, 1],
                       c=colour, s=20, alpha=0.75, label=f"H{dim}", zorder=3)

        lim = max(all_finite, default=1.0) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.4)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(label)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_aspect("equal")

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Optional[str | Path]) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from tda_fruit_ct.synthetic import sphere_with_cavities, filtration_field
    from tda_fruit_ct.filtration import sublevel_filtration
    from tda_fruit_ct.persistence import compute_persistence
    from tda_fruit_ct.vectorization import betti_curve, persistence_image

    print("Generating phantom and running TDA for viz smoke test...")
    vol, n_cav = sphere_with_cavities(shape=(32, 32, 32), n_cavities=2)
    diagrams = compute_persistence(sublevel_filtration(filtration_field(vol)))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        plot_slices(vol, title="Phantom slices", save_path=td / "slices.png")
        print("  plot_slices OK")

        plot_persistence_diagram(diagrams, title="Phantom PD", save_path=td / "pd.png")
        print("  plot_persistence_diagram OK")

        betti_data = {dim: betti_curve(diag) for dim, diag in diagrams.items()}
        plot_betti_curves(betti_data, title="Phantom Betti", save_path=td / "betti.png")
        print("  plot_betti_curves OK")

        pi = persistence_image(diagrams[2], resolution=(20, 20))
        plot_persistence_image(pi, title="H₂ PI", save_path=td / "pi.png")
        print("  plot_persistence_image OK")

        plot_diagram_comparison(diagrams, diagrams, labels=("A", "B"), save_path=td / "cmp.png")
        print("  plot_diagram_comparison OK")

    plt.close("all")
    print("PASS")
