import argparse
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference using trained meta model.")
    parser.add_argument("--model", default="artifacts/meta_model.joblib")
    parser.add_argument("--features", default="dataset/meta_features.csv")
    parser.add_argument("--output", default="dataset/meta_predictions.csv")
    parser.add_argument("--metrics-scope", choices=("holdout", "all", "none"), default="holdout")
    args, _unknown = parser.parse_known_args()

    try:
        artifact = joblib.load(args.model)
    except (AttributeError, ModuleNotFoundError, OSError) as e:
        print(
            "Failed to load meta_model.joblib.\n"
            "This usually happens when the model was trained on a different OS/Python "
            "or a different scikit-learn build (common: Kaggle Linux -> Windows local).\n"
            "HistGradientBoostingRegressor pickles are not reliably portable across platforms.\n\n"
            "Fix: retrain on this machine using the same feature CSV:\n"
            "  python -m meta_model.train --features dataset/meta_features.csv "
            "--model-out artifacts/meta_model.joblib --report-out artifacts/meta_model_report.json\n"
            "(add --strict-features if you trained that way on Kaggle)\n\n"
            f"Local sklearn: {sklearn.__version__}, Python: {sys.version.split()[0]}, OS: {platform.platform()}\n"
            f"Original error: {e!r}"
        )
        raise SystemExit(1) from e

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    training_mode = artifact.get("training_mode", "absolute")
    df = pd.read_csv(args.features)

    # Verify feature consistency
    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        print(
            f"WARNING: feature mismatch — columns {missing_cols} missing from CSV. "
            f"Falling back to engine score (raw_score_10)."
        )
        preds = np.clip(df["raw_score_10"].astype(float).values, 1.0, 10.0)
    else:
        X = df[feature_columns].fillna(0.0).astype(float).values
        if training_mode == "residual":
            engine_scores = df["raw_score_10"].astype(float).values
            delta = model.predict(X)
            clip_val = artifact.get("residual_delta_clip", 1.8)
            preds = np.clip(
                engine_scores + np.clip(delta, -clip_val, clip_val), 1.0, 10.0
            )
        else:
            preds = np.clip(model.predict(X), 1.0, 10.0)

    out = df.copy()
    out["meta_pred_score"] = preds
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote predictions for all {len(out)} rows to: {out_path}")

    if args.metrics_scope == "none" or "human_score" not in out.columns:
        return

    y_all = out["human_score"].astype(float).values
    test_idx = artifact.get("test_indices")
    if args.metrics_scope == "holdout":
        if not test_idx:
            raise SystemExit(
                "ERROR: model artifact has no test_indices (older/incompatible training run). "
                "Re-run meta training to embed holdout row ids."
            )
        idx = np.array(test_idx, dtype=int)
        if idx.max() >= len(out) or idx.min() < 0:
            raise SystemExit(
                "ERROR: test_indices from model do not match this features file "
                "(regenerate meta_features.csv and retrain)."
            )
        y = y_all[idx]
        p_hat = preds[idx]
        scope_note = f"holdout test rows only (n={len(idx)})"
    else:
        y = y_all
        p_hat = preds
        scope_note = "ALL rows (includes rows the model trained on — optimistic / not a fair test)"

    p, _ = pearsonr(y, p_hat)
    s, _ = spearmanr(y, p_hat)
    mae = float(mean_absolute_error(y, p_hat))
    print(f"\nMetrics ({scope_note}):")
    print(f"Pearson  : {p:.3f}")
    print(f"Spearman : {s:.3f}")
    print(f"MAE      : {mae:.3f}")


if __name__ == "__main__":
    main()
