import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error

from meta_model.split_utils import fixed_split_indices, load_question_split


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p, _ = pearsonr(y_true, y_pred)
    s, _ = spearmanr(y_true, y_pred)
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"pearson": float(p), "spearman": float(s), "mae": mae}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen unseen-question benchmark gate.")
    parser.add_argument("--features", default="dataset/meta_features.csv")
    parser.add_argument("--model", default="artifacts/meta_model.joblib")
    parser.add_argument("--split-file", default="artifacts/question_holdout_split.json")
    parser.add_argument("--report-out", default="artifacts/unseen_benchmark_report.json")
    args, _unknown = parser.parse_known_args()

    df = pd.read_csv(args.features)
    if "question_id" not in df.columns or "human_score" not in df.columns:
        raise ValueError("Feature CSV must include question_id and human_score.")

    _, test_ids = load_question_split(args.split_file)
    train_idx, test_idx = fixed_split_indices(df, args.split_file)
    test_questions_seen = sorted(df.iloc[test_idx]["question_id"].astype(str).unique().tolist())
    if sorted(test_ids) != sorted(test_questions_seen):
        raise ValueError("Frozen split mismatch: feature file does not align with split-file test IDs.")

    artifact = joblib.load(args.model)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    X = df[feature_columns].fillna(0.0).astype(float).values
    y = df["human_score"].astype(float).values
    baseline = np.clip(df["raw_score_10"].astype(float).values, 1.0, 10.0)
    pred = np.clip(model.predict(X), 1.0, 10.0)

    y_test = y[test_idx]
    pred_test = pred[test_idx]
    base_test = baseline[test_idx]
    report = {
        "split_file": args.split_file,
        "rows_total": int(len(df)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "frozen_test_question_ids": sorted(test_ids),
        "baseline_test_metrics": _metrics(y_test, base_test),
        "meta_model_test_metrics": _metrics(y_test, pred_test),
    }

    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== Frozen unseen-question benchmark ===")
    print(f"Split file: {args.split_file}")
    print(f"Rows train/test: {len(train_idx)}/{len(test_idx)}")
    print("Baseline:", report["baseline_test_metrics"])
    print("MetaModel:", report["meta_model_test_metrics"])
    print(f"Saved report to: {out}")


if __name__ == "__main__":
    main()
