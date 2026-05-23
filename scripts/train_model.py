"""Train a ML model to predict browning score from TDA features.

With only 9 samples we use Leave-One-Out cross-validation to get
honest performance estimates.

Usage:
    python scripts/train_model.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / "results" / "tda_features.csv"
RESULTS_DIR = ROOT / "results"


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} samples, {len(df.columns)} columns")

    # Target
    y = df["browning_score"].values.astype(float)
    apple_ids = df["apple_id"].astype(str).tolist()

    # Features: tout sauf apple_id et browning_score
    feature_cols = [c for c in df.columns if c not in ("apple_id", "browning_score")]

    # On garde deux jeux de features:
    # 1. Features interpretables (comptes + persistences)
    topo_cols = [c for c in feature_cols if any(
        c.startswith(p) for p in ("h0_", "h1_", "h2_")
    )]
    # 2. Features completes (PI + Betti curves + topo)
    all_cols = feature_cols

    return df, y, apple_ids, topo_cols, all_cols


def loo_evaluate(X: np.ndarray, y: np.ndarray, model, name: str) -> dict:
    """Leave-One-Out cross-validation."""
    loo = LeaveOneOut()
    y_pred = np.zeros_like(y)

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        model.fit(X_train, y_train)
        y_pred[test_idx] = model.predict(X_test)

    mae = mean_absolute_error(y, y_pred)
    # R2 peut être negatif avec LOO sur peu de samples, c'est normal
    r2 = r2_score(y, y_pred)

    print(f"\n  {name}")
    print(f"    MAE  : {mae:.2f} (browning score points)")
    print(f"    R²   : {r2:.3f}")
    print(f"    True : {y.tolist()}")
    print(f"    Pred : {[round(p, 1) for p in y_pred.tolist()]}")

    return {"model": name, "mae": mae, "r2": r2, "y_true": y, "y_pred": y_pred}


def main():
    if not FEATURES_CSV.exists():
        print(f"Features not found: {FEATURES_CSV}")
        print("Run: python scripts/run_pipeline.py")
        return

    df, y, apple_ids, topo_cols, all_cols = load_features(FEATURES_CSV)

    print(f"\nTarget (browning scores): {y.tolist()}")
    print(f"Topological features: {len(topo_cols)} columns")
    print(f"Full features (PI+BC+topo): {len(all_cols)} columns")

    # ----------------------------------------------------------------
    # Jeu 1 : features topologiques interpretables seulement
    # ----------------------------------------------------------------
    print("\n" + "="*55)
    print("MODELES SUR FEATURES TOPOLOGIQUES (interpretables)")
    print("="*55)
    X_topo = df[topo_cols].values.astype(float)

    results = []
    for name, model in [
        ("Ridge regression", Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])),
        ("Random Forest", RandomForestRegressor(
            n_estimators=100, max_depth=3, random_state=42
        )),
        ("Gradient Boosting", GradientBoostingRegressor(
            n_estimators=50, max_depth=2, random_state=42
        )),
    ]:
        r = loo_evaluate(X_topo, y, model, name)
        results.append(r)

    # ----------------------------------------------------------------
    # Jeu 2 : features completes (PI + Betti curves)
    # ----------------------------------------------------------------
    print("\n" + "="*55)
    print("MODELES SUR FEATURES COMPLETES (PI + Betti curves)")
    print("="*55)
    X_full = df[all_cols].values.astype(float)

    for name, model in [
        ("Ridge regression (full)", Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ])),
        ("Random Forest (full)", RandomForestRegressor(
            n_estimators=100, max_depth=2, random_state=42
        )),
    ]:
        r = loo_evaluate(X_full, y, model, name)
        results.append(r)

    # ----------------------------------------------------------------
    # Meilleur modele + analyse
    # ----------------------------------------------------------------
    best = min(results, key=lambda r: r["mae"])
    print("\n" + "="*55)
    print(f"MEILLEUR MODELE : {best['model']}")
    print(f"  MAE = {best['mae']:.2f} points de browning")
    print(f"  R²  = {best['r2']:.3f}")
    print("="*55)

    # Tableau recapitulatif
    summary = pd.DataFrame({
        "apple_id": apple_ids,
        "browning_score_reel": y,
        "browning_score_predit": [round(p, 1) for p in best["y_pred"]],
        "erreur": [round(abs(p - t), 1) for p, t in zip(best["y_pred"], y)],
        "h2_cavites": df["h2_significant"].values,
        "h2_max_persistence": df["h2_max_persistence"].round(3).values,
    })

    print("\nTableau recapitulatif :")
    print(summary.to_string(index=False))

    # Sauvegarder
    out = RESULTS_DIR / "model_predictions.csv"
    summary.to_csv(out, index=False)
    print(f"\nPredictions sauvegardees : {out}")

    # ----------------------------------------------------------------
    # Interpretation : quelles features comptent le plus ?
    # ----------------------------------------------------------------
    print("\n" + "="*55)
    print("INTERPRETATION — correlation features vs browning")
    print("="*55)
    corr_data = []
    for col in topo_cols:
        vals = df[col].values.astype(float)
        if vals.std() > 0:
            corr = float(np.corrcoef(vals, y)[0, 1])
            corr_data.append({"feature": col, "correlation": round(corr, 3)})

    corr_df = pd.DataFrame(corr_data).sort_values("correlation", key=abs, ascending=False)
    print(corr_df.head(10).to_string(index=False))
    print("\nCorrelation positive = feature augmente avec le browning")
    print("Correlation negative = feature diminue avec le browning")


if __name__ == "__main__":
    main()
