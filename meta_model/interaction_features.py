"""Hand-crafted interaction columns appended at train/inference — numpy only."""

import numpy as np


def add_interaction_features(X: np.ndarray, feature_columns: list) -> np.ndarray:
    col = {name: i for i, name in enumerate(feature_columns)}
    extras = []
    if "concept_score" in col and "coverage_06" in col:
        extras.append(X[:, col["concept_score"]] * X[:, col["coverage_06"]])
    if "adj_min" in col and "adj_mean" in col:
        extras.append(X[:, col["adj_min"]] * X[:, col["adj_mean"]])
    if "combined_min" in col and "combined_std" in col:
        extras.append(X[:, col["combined_min"]] / (X[:, col["combined_std"]] + 0.01))
    if "zero_credit_ratio" in col and "concept_score" in col:
        extras.append(X[:, col["zero_credit_ratio"]] * X[:, col["concept_score"]])
    if "raw_sim_min" in col:
        extras.append(np.clip(X[:, col["raw_sim_min"]] - 0.55, 0.0, None))
    if extras:
        return np.hstack([X, np.column_stack(extras)])
    return X
