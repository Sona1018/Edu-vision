from meta_model.validate import main
import numpy as np
import pandas as pd
import argparse
import json

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p, _ = pearsonr(y_true, y_pred)
    s, _ = spearmanr(y_true, y_pred)
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"pearson": float(p), "spearman": float(s), "mae": mae}


def _summarize(rows: list[dict]) -> dict:
    keys = ["pearson", "spearman", "mae"]
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}
    return out


def _band_error_breakdown(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    bands = [(1, 3), (4, 6), (7, 8), (9, 10)]
    out = {}
    for lo, hi in bands:
        mask = (y_true >= lo) & (y_true <= hi)
        key = f"{lo}-{hi}"
        if int(mask.sum()) == 0:
            out[key] = {"n": 0, "mae": None}
            continue
        mae = float(mean_absolute_error(y_true[mask], y_pred[mask]))
        out[key] = {"n": int(mask.sum()), "mae": mae}
    return out


def _per_question_error(df_test: pd.DataFrame, y_pred: np.ndarray) -> list[dict]:
    rows = []
    for qid, part in df_test.groupby("question_id"):
        y_true = part["human_score"].astype(float).values
        idx = part.index.to_numpy()
        p = y_pred[idx]
        rows.append(
            {
                "question_id": str(qid),
                "n": int(len(part)),
                "mae": float(mean_absolute_error(y_true, p)),
            }
        )
    rows.sort(key=lambda r: r["mae"], reverse=True)
    return rows


def _leakage_audit(df: pd.DataFrame, feature_columns: list[str]) -> dict:
    y = df["human_score"].astype(float).values
    risk_flags = []

    # 1) suspicious name pattern audit
    suspicious_tokens = ("human", "target", "label", "score_final", "calibrated")
    suspicious_cols = [
        c for c in feature_columns if any(tok in c.lower() for tok in suspicious_tokens)
    ]
    if suspicious_cols:
        risk_flags.append(f"suspicious_feature_names={suspicious_cols}")

    # 2) near-perfect single-feature correlation with target
    corr_rows = []
    for c in feature_columns:
        x = df[c].astype(float).values
        if np.std(x) < 1e-12:
            continue
        p, _ = pearsonr(x, y)
        corr_rows.append((c, float(abs(p))))
    corr_rows.sort(key=lambda t: t[1], reverse=True)
    top_corr = corr_rows[:8]
    if top_corr and top_corr[0][1] > 0.98:
        risk_flags.append(f"near_perfect_feature_corr={top_corr[0]}")

    return {"risk_flags": risk_flags, "top_abs_feature_correlations": top_corr}


def run_group_cv(
    df: pd.DataFrame,
    feature_columns: list[str],
    group_values: np.ndarray,
    n_splits: int,
    random_seed: int,
) -> dict:
    X = df[feature_columns].fillna(0.0).astype(float).values
    y = df["human_score"].astype(float).values
    baseline = np.clip(df["raw_score_10"].astype(float).values, 1.0, 10.0)

    gkf = GroupKFold(n_splits=n_splits)
    baseline_rows: list[dict] = []
    model_rows: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=group_values), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        base_test = baseline[test_idx]

        model = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=450,
            min_samples_leaf=8,
            l2_regularization=0.05,
            random_state=random_seed + fold_idx,
        )
        model.fit(X_train, y_train)
        pred = np.clip(model.predict(X_test), 1.0, 10.0)

        b = _metrics(y_test, base_test)
        m = _metrics(y_test, pred)
        b["fold"] = fold_idx
        m["fold"] = fold_idx
        baseline_rows.append(b)
        model_rows.append(m)

    return {
        "baseline_per_fold": baseline_rows,
        "meta_model_per_fold": model_rows,
        "baseline_summary": _summarize(baseline_rows),
        "meta_model_summary": _summarize(model_rows),
    }


def run_fixed_holdout_eval(
    df: pd.DataFrame,
    feature_columns: list[str],
    random_seed: int,
    split_file: str,
) -> dict:
    X = df[feature_columns].fillna(0.0).astype(float).values
    y = df["human_score"].astype(float).values
    baseline = np.clip(df["raw_score_10"].astype(float).values, 1.0, 10.0)
    train_idx, test_idx = fixed_split_indices(df, split_file)

    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=450,
        min_samples_leaf=8,
        l2_regularization=0.05,
        random_state=random_seed,
    )
    model.fit(X[train_idx], y[train_idx])
    pred = np.clip(model.predict(X[test_idx]), 1.0, 10.0)
    base_test = baseline[test_idx]
    y_test = y[test_idx]
    return {
        "split_file": split_file,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "baseline_test_metrics": _metrics(y_test, base_test),
        "meta_model_test_metrics": _metrics(y_test, pred),
        "score_band_mae_baseline": _band_error_breakdown(y_test, base_test),
        "score_band_mae_meta_model": _band_error_breakdown(y_test, pred),
        "per_question_mae_meta_model": _per_question_error(
            df.loc[test_idx, ["question_id", "human_score"]],
            pd.Series(pred, index=test_idx).reindex(df.index, fill_value=np.nan).values,
        ),
    }


def build_cluster_groups(df: pd.DataFrame, random_seed: int) -> np.ndarray:
    # Strict secondary split: cluster by answer shape + rubric size, then group by cluster.
    cluster_feats = [
        "answer_len_words",
        "answer_len_chars",
        "n_concepts",
        "total_weight",
        "semantic_score",
        "structural_score",
    ]
    arr = df[cluster_feats].fillna(0.0).astype(float).values
    n_clusters = min(8, max(3, len(df) // 40))
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_seed)
    return kmeans.fit_predict(arr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict CV validation for IntelliPlace meta-model.")
    parser.add_argument("--features", default="dataset/meta_features.csv")
    parser.add_argument("--report-out", default="artifacts/meta_cv_report.json")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--split-file",
        default="artifacts/question_holdout_split.json",
        help="Fixed question-level holdout split JSON for reproducible holdout reporting.",
    )
    parser.add_argument(
        "--strict-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Exclude score-proxy features (raw_score_10, hybrid_score_100). "
            "Default is strict mode ON; pass --no-strict-features to include proxies."
        ),
    )
    parser.add_argument(
        "--with-cluster-split",
        action="store_true",
        help="Add a stricter second CV pass grouped by answer-shape clusters.",
    )
    args, _unknown = parser.parse_known_args()

    df = pd.read_csv(args.features)
    feature_columns = [
        c for c in ALL_FEATURE_COLUMNS if (c not in PROXY_FEATURES or not args.strict_features)
    ]
    required = set(feature_columns + ["human_score", "question_id", "raw_score_10"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in feature CSV: {missing}")

    leakage = _leakage_audit(df, feature_columns)
    q_groups = df["question_id"].astype(str).values
    by_question = run_group_cv(
        df=df,
        feature_columns=feature_columns,
        group_values=q_groups,
        n_splits=args.n_splits,
        random_seed=args.random_seed,
    )
    fixed_holdout = run_fixed_holdout_eval(
        df=df,
        feature_columns=feature_columns,
        random_seed=args.random_seed,
        split_file=args.split_file,
    )

    report = {
        "rows_total": int(len(df)),
        "feature_count": len(feature_columns),
        "strict_features": bool(args.strict_features),
        "feature_columns": feature_columns,
        "leakage_audit": leakage,
        "group_by_question_id_cv": by_question,
        "fixed_question_holdout": fixed_holdout,
    }

    if args.with_cluster_split:
        cluster_groups = build_cluster_groups(df, args.random_seed)
        cluster_cv = run_group_cv(
            df=df,
            feature_columns=feature_columns,
            group_values=cluster_groups,
            n_splits=min(args.n_splits, len(np.unique(cluster_groups))),
            random_seed=args.random_seed + 99,
        )
        report["group_by_answer_cluster_cv"] = cluster_cv

    out_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    b = by_question["baseline_summary"]
    m = by_question["meta_model_summary"]
    print("=== GroupKFold by question_id ===")
    print(f"Baseline Pearson mean±std : {b['pearson']['mean']:.3f} ± {b['pearson']['std']:.3f}")
    print(f"Meta     Pearson mean±std : {m['pearson']['mean']:.3f} ± {m['pearson']['std']:.3f}")
    print(f"Baseline Spearman mean±std: {b['spearman']['mean']:.3f} ± {b['spearman']['std']:.3f}")
    print(f"Meta     Spearman mean±std: {m['spearman']['mean']:.3f} ± {m['spearman']['std']:.3f}")
    print(f"Baseline MAE mean±std     : {b['mae']['mean']:.3f} ± {b['mae']['std']:.3f}")
    print(f"Meta     MAE mean±std     : {m['mae']['mean']:.3f} ± {m['mae']['std']:.3f}")
    print("\n=== Fixed question holdout ===")
    print(f"Split file: {fixed_holdout['split_file']}")
    print(f"Rows train/test: {fixed_holdout['train_rows']}/{fixed_holdout['test_rows']}")
    print(f"Baseline: {fixed_holdout['baseline_test_metrics']}")
    print(f"MetaModel: {fixed_holdout['meta_model_test_metrics']}")
    print(f"Saved report to: {out_path}")


if __name__ == "__main__":
    main()
