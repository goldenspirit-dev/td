# tda-fruit-ct

> **Status:** private research repository.

Topological Data Analysis of fruit CT scans using persistent homology (GUDHI).

## Research questions

1. **Cavity detection** — Use H₂ persistent homology on 3-D CT volumes to detect and
   characterise internal cavities, voids, and browning-related defects in fruit.
2. **Fruit-type classification** — Vectorise persistence diagrams (persistence images,
   Betti curves, landscapes) and train classifiers on the TDA features.
3. **Ripeness & storage indicators** — Track how topological features evolve with
   storage conditions to find TDA-based quality indicators.

## Quickstart

### 1 — Set up the environment

```powershell
# Create virtual environment (Python 3.11)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

> **Note on numpy / scikit-learn compatibility:**  
> After installing all dependencies, pin numpy to 1.26.4 to avoid an ABI clash
> between scikit-learn 1.3.x (required by giotto-tda) and newer numpy:
> ```powershell
> pip install "numpy==1.26.4" --force-reinstall
> ```

### 2 — Download data

```powershell
# Option A: Download a small sanity-check CT volume (~16 MB)
python src/download_data.py --dataset scivis

# Option B: Download Kanzi apple metadata only (browning_scores.csv + label image)
python src/download_data.py --dataset kanzi

# Option C: Download the full 103 GB Kanzi CT ZIP (requires manual confirmation)
python src/download_data.py --dataset kanzi --full
```

### 3 — Run the EDA notebook

```powershell
jupyter lab
# Then open: notebooks/01_eda_and_first_inference.ipynb
# Select kernel: Python 3 (.venv)
```

The notebook works without any real CT data — it falls back to a synthetic phantom
for all pipeline sections.  Download the SciVis bonsai for a real-volume demo.

## Project structure

```
tda-fruit-ct/
|-- data/                      # (gitignored) raw / interim / processed volumes
|   |-- raw/kanzi_apples/      # browning_scores.csv, slicing_machine_labels.png
|   +-- raw/scivis/            # bonsai_256x256x256_uint8.raw (after download)
|-- notebooks/
|   +-- 01_eda_and_first_inference.ipynb   # full EDA pipeline
|-- src/
|   |-- download_data.py       # download script (standalone, run directly)
|   +-- tda_fruit_ct/          # importable package
|       |-- synthetic.py       # phantom generator (sphere + N cavities)
|       |-- data_loader.py     # load .npy / .raw / .nii / .mhd volumes
|       |-- filtration.py      # GUDHI cubical complex builder + downsample
|       |-- persistence.py     # compute diagrams, filter noise, summary
|       |-- vectorization.py   # persistence images, Betti curves, landscapes
|       |-- viz.py             # plotting: slices, PD, Betti curves, PI heatmaps
|       |-- preprocessing.py   # (stub) denoise, resample, segment
|       +-- classification.py  # (stub) ML on TDA features
|-- results/figures/           # output plots (gitignored)
|-- configs/default.yaml       # YAML run config
|-- scripts/compute_persistence.py  # (stub) batch CLI
+-- tests/                     # unit tests
```

## Module overview

| Module | Purpose |
|--------|---------|
| `synthetic.py` | Generate sphere-with-N-cavities and torus phantoms for validation |
| `data_loader.py` | Load CT volumes (.npy, .raw, NIfTI, MHD); auto-normalise to float32 in [0,1] |
| `filtration.py` | Build GUDHI CubicalComplex; downsample, smooth, threshold helpers |
| `persistence.py` | Compute H₀/H₁/H₂ diagrams; filter noise; count significant features |
| `vectorization.py` | Persistence images (20×20), Betti curves (100 pts), landscapes |
| `viz.py` | Slice viewer, persistence diagram scatter, Betti curve plots, PI heatmaps |
| `src/download_data.py` | Download Kanzi apple dataset from Zenodo or SciVis bonsai |

## Datasets

| Dataset | Size | Notes |
|---------|------|-------|
| **Kanzi apple CT** (Zenodo 8167285) | 103 GB ZIP | 120 apples, browning scores, FDK reconstructions |
| **Open SciVis bonsai** | ~16 MB | 256³ uint8 raw binary; used as quick sanity check |
| **Synthetic phantoms** | <1 MB | Generated locally; ground-truth topology known |

Data is **not tracked by git** — see `data/README.md`.

## Key result (pipeline validation)

On a 64³ synthetic phantom with **N = 4 spherical cavities**:

```
H0 (connected components) : 1  (one solid ball)        -- PASS
H1 (loops/tunnels)        : 0  (no tunnels)            -- PASS
H2 (enclosed cavities)    : 4  (one per void)          -- PASS
```

Top H₂ features have persistence ≈ 0.97 (birth ≈ 0.004, death ≈ 0.979),
cleanly separated from sub-threshold noise.

## License

TBD.
