"""Run the full TDA pipeline on the reduced Kanzi apple dataset.

For each apple in data/interim/reduced/:
  1. Load the .npz volume
  2. Downsample to 64x64x64
  3. Smooth + build filtration
  4. Compute H0/H1/H2 persistence
  5. Vectorise (persistence images + Betti curves)
  6. Save results to results/tda_features.csv

Usage:
    python scripts/run_pipeline.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from tda_fruit_ct import filtration, persistence, vectorization, synthetic

ROOT        = Path(__file__).resolve().parent.parent
REDUCED_DIR = ROOT / "data" / "interim" / "reduced"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SHAPE = (64, 64, 64)
MIN_PERSISTENCE = 0.05


def process_apple(npz_path: Path) -> dict:
    apple_id = npz_path.stem.replace("apple_", "")
    print(f"\n── Apple {apple_id} ──")

    # 1. Load
    vol = np.load(npz_path)["volume"].astype(np.float32)
    print(f"  Loaded: shape={vol.shape}")

    # 2. Downsample
    vol = filtration.downsample(vol, TARGET_SHAPE)
    print(f"  Downsampled to {vol.shape}")

    # 3. Smooth + filtration field (1 - density, solid first)
    vol = filtration.smooth(vol, sigma=0.8)
    field = 1.0 - vol

    # 4. Persistence
    t0 = time.time()
    cc = filtration.sublevel_filtration(field)
    diags = persistence.compute_persistence(cc, min_persistence=0.0)
    elapsed = time.time() - t0
    print(f"  Persistence computed in {elapsed:.1f}s")
    print(f"  {persistence.summary(diags, min_persistence=MIN_PERSISTENCE)}")

    # 5. Vectorise
    vecs = vectorization.vectorize_diagrams(
        diags, dims=(0, 1, 2),
        pi_resolution=(20, 20), bc_resolution=100, bandwidth=0.05,
    )

    # 6. Build result row
    row = {"apple_id": apple_id}

    # Significant feature counts
    for dim in (0, 1, 2):
        row[f"h{dim}_significant"] = persistence.count_significant(diags[dim], MIN_PERSISTENCE)
        row[f"h{dim}_total"] = len(diags[dim])
        finite = persistence.finite_diagram(diags[dim])
        if len(finite) > 0:
            pers = finite[:, 1] - finite[:, 0]
            row[f"h{dim}_max_persistence"] = float(pers.max())
            row[f"h{dim}_mean_persistence"] = float(pers.mean())
        else:
            row[f"h{dim}_max_persistence"] = 0.0
            row[f"h{dim}_mean_persistence"] = 0.0

    # Flatten feature vectors
    for key, vec in vecs.items():
        for i, val in enumerate(vec):
            row[f"{key}_{i}"] = float(val)

    return row


def main():
    npz_files = sorted(REDUCED_DIR.glob("apple_*.npz"))
    if not npz_files:
        print(f"No .npz files found in {REDUCED_DIR}")
        print("Run: python src/reduce_dataset.py --n-per-group 3")
        return

    # Load browning scores for context
    scores_path = ROOT / "data" / "raw" / "kanzi_apples" / "browning_scores.csv"
    scores_df = None
    if scores_path.exists():
        scores_df = pd.read_csv(scores_path)
        scores_df.columns = [c.strip().lower().replace(" ", "_") for c in scores_df.columns]

    print(f"Found {len(npz_files)} apples to process")
    print(f"Target shape: {TARGET_SHAPE}")

    rows = []
    for npz_path in npz_files:
        try:
            row = process_apple(npz_path)
            rows.append(row)
        except Exception as e:
            print(f"  ERROR on {npz_path.name}: {e}")

    # Save features
    df = pd.DataFrame(rows)

    # Merge browning scores if available
    if scores_df is not None:
        score_col = next((c for c in scores_df.columns if "browning" in c or "score" in c), None)
        id_col = next((c for c in scores_df.columns if "id" in c or "apple" in c), scores_df.columns[0])
        if score_col:
            scores_df = scores_df.rename(columns={id_col: "apple_id", score_col: "browning_score"})
            scores_df["apple_id"] = scores_df["apple_id"].astype(str)
            df["apple_id"] = df["apple_id"].astype(str)
            df = df.merge(scores_df[["apple_id", "browning_score"]], on="apple_id", how="left")

    out_path = RESULTS_DIR / "tda_features.csv"
    df.to_csv(out_path, index=False)

    # Print summary
    print(f"\n{'='*50}")
    print(f"Done! {len(rows)} apples processed.")
    print(f"Features saved to: {out_path}")
    print(f"\nSummary (topological features per apple):")
    summary_cols = ["apple_id"] + (["browning_score"] if "browning_score" in df.columns else []) + \
                   ["h0_significant", "h1_significant", "h2_significant",
                    "h2_max_persistence", "h2_mean_persistence"]
    print(df[summary_cols].to_string(index=False))
    print(f"\nTotal feature vector size: {len(df.columns)} columns")


if __name__ == "__main__":
    main()
