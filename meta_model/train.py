import argparse
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

try:
    from meta_model.split_utils import fixed_split_indices
except ModuleNotFoundError:
    from split_utils import fixed_split_indices

try:
    from meta_model.interaction_features import add_interaction_features
except ModuleNotFoundError:  # pragma: no cover
    from interaction_features import add_interaction_features


# Max residual correction applied at inference (must match saved artifact after training).
DEFAULT_RESIDUAL_DELTA_CLIP = 2.35

ALL_FEATURE_COLUMNS = [
    # Length
    "answer_len_words",
    "answer_len_chars",
    "n_concepts",
    "total_weight",
    # Semantic — off_topic_gate excluded (constant=1.0, zero variance)
    "semantic_score",
    # Core engine scores
    "concept_score",
    "gated_concept_score",
    "structural_score",
    "raw_score_10",
    # Per-concept similarity — adj_min is the strongest discriminator
    "adj_mean",
    "adj_max",
    "adj_min",
    "adj_std",
    "adj_range",
    # Coverage
    "coverage_03",
    "coverage_04",
    "coverage_06",
    # Zero-credit concepts
    "zero_credit_count",
    "zero_credit_ratio",
    # NLI signals
    "contradiction_count",
    "contradiction_mean",
    "contradiction_max",
    "entailment_mean",
    "entailment_max",
    "entailment_min",
    # Depth
    "depth_mean",
    "depth_max",
    # Combined quality — combined_min is a strong discriminator
    "combined_mean",
    "combined_max",
    "combined_min",
    "combined_std",
    # Raw similarity variance
    "sim_std_mean",
    "sim_std_max",
    "raw_sim_mean",
    "raw_sim_max",
    "raw_sim_min",
    "raw_sim_std",
    # Derived concept consistency / weakness (from prepare_features.py)
    "consistency_score",
    "weak_signal",
    "concept_range",
    "dominance_score",
    "consistency_ratio",
    "contradiction_score",
    "contradiction_severity",
]

# Proxy features excluded in strict mode (too correlated with target)
PROXY_FEATURES = {"raw_score_10", "hybrid_score_100"}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p, _ = pearsonr(y_true, y_pred)
    s, _ = spearmanr(y_true, y_pred)
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"pearson": float(p), "spearman": float(s), "mae": mae}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train meta scoring model on feature CSV.")
    parser.add_argument("--features",    default="dataset/meta_features.csv")
    parser.add_argument("--model-out",   default="artifacts/meta_model.joblib")
    parser.add_argument("--report-out",  default="artifacts/meta_model_report.json")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--split-mode",  choices=("fixed", "random"), default="fixed")
    parser.add_argument("--split-file",  default="artifacts/question_holdout_split.json")
    parser.add_argument(
        "--strict-features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args, _unknown = parser.parse_known_args()

    df = pd.read_csv(args.features)

    # Only keep columns that exist in the CSV and pass the strict filter
    feature_columns = [
        c for c in ALL_FEATURE_COLUMNS
        if c in df.columns
        and (c not in PROXY_FEATURES or not args.strict_features)
    ]

    required = set(feature_columns + ["human_score", "question_id"])
    missing  = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in feature CSV: {missing}")

    X_raw  = df[feature_columns].fillna(0.0).astype(float).values
    X      = add_interaction_features(X_raw, feature_columns)
    y      = df["human_score"].astype(float).values
    groups = df["question_id"].astype(str).values
    baseline = df["raw_score_10"].astype(float).values

    if args.split_mode == "fixed":
        train_idx, test_idx = fixed_split_indices(df, args.split_file)
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.random_seed)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    train_idx = np.array(train_idx, dtype=int)
    test_idx  = np.array(test_idx,  dtype=int)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    baseline_test   = baseline[test_idx]

    model = StackingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=200, random_state=42)),
            ("gbr", GradientBoostingRegressor(n_estimators=200, random_state=42)),
        ],
        final_estimator=Ridge(),
    )

    # Train on residuals
    residuals_train = y_train - baseline[train_idx]
    model.fit(X_train, residuals_train)

    pred_residuals = model.predict(X_test)
    pred_residuals = np.clip(pred_residuals, -DEFAULT_RESIDUAL_DELTA_CLIP, DEFAULT_RESIDUAL_DELTA_CLIP)
    pred_test = np.clip(baseline_test + pred_residuals, 1.0, 10.0)
    base_test = np.clip(baseline_test, 1.0, 10.0)

    model_metrics    = _metrics(y_test, pred_test)
    baseline_metrics = _metrics(y_test, base_test)
    test_indices     = test_idx.astype(int).tolist()

    artifact = {
        "model":           model,
        "feature_columns": feature_columns,
        "training_mode":   "residual",
        "residual_delta_clip": DEFAULT_RESIDUAL_DELTA_CLIP,
        "train_size":      int(len(train_idx)),
        "test_size":       int(len(test_idx)),
        "test_indices":    test_indices,
        "random_seed":     args.random_seed,
        "split_mode":      args.split_mode,
        "split_file":      args.split_file if args.split_mode == "fixed" else None,
        "strict_features": bool(args.strict_features),
        "sklearn_version": sklearn.__version__,
        "python_version":  sys.version.split()[0],
        "platform":        platform.platform(),
    }
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    report = {
        "baseline_test_metrics":   baseline_metrics,
        "meta_model_test_metrics": model_metrics,
        "rows_total":   int(len(df)),
        "train_rows":   int(len(train_idx)),
        "test_rows":    int(len(test_idx)),
        "test_indices": test_indices,
        "feature_columns": feature_columns,
        "split_mode":   args.split_mode,
        "split_file":   args.split_file if args.split_mode == "fixed" else None,
        "strict_features": bool(args.strict_features),
        "sklearn_version": sklearn.__version__,
        "python_version":  sys.version.split()[0],
        "platform":        platform.platform(),
    }
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Baseline :", baseline_metrics)
    print("MetaModel:", model_metrics)
    print(f"Saved model  → {model_path}")
    print(f"Saved report → {report_path}")


if __name__ == "__main__":
    main()
