"""
metrics_quick.py
-----------------
Compute final correlation metrics for the current dataset/pipeline,
without per-answer logging spam.

This does NOT modify any model/code used by the engine.
It is only a verification runner.
"""

import json
import numpy as np
from scipy.stats import pearsonr, spearmanr

from engine.concept_detection import compute_concept_score
from engine.mastery import (
    compute_contradiction_ratio,
    compute_hybrid_score,
    compute_structural_score,
    soft_coverage_ratio,
)
from engine.scoring import compute_semantic_score


def calibrate_scores(
    raw_scores: np.ndarray,
    human_scores: np.ndarray | None = None,
    target_min: float = 1.0,
    target_max: float = 10.0,
) -> np.ndarray:
    raw = np.array(raw_scores, dtype=float)
    r_min = raw.min()
    r_max = raw.max()

    if r_max == r_min:
        return np.full_like(raw, (target_min + target_max) / 2.0)

    stretched = (raw - r_min) / (r_max - r_min)
    stretched = stretched * (target_max - target_min) + target_min

    if human_scores is not None:
        bias = float(np.mean(human_scores, dtype=float) - np.mean(stretched, dtype=float))
        stretched = np.clip(stretched + bias, target_min, target_max)

    return stretched


def main() -> None:
    with open("dataset/intelliplace_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    human_scores: list[float] = []
    raw_engine_scores: list[float] = []

    for q in dataset:
        rubric = q["rubric"]
        total_weight = float(sum(c["weight"] for c in rubric))
        combined_reference = " ".join(c["reference_explanation"] for c in rubric)

        for ans in q["answers"]:
            student_answer = ans["text"]
            human_score = ans["score"]

            semantic_score, off_topic_gate = compute_semantic_score(
                student_answer, combined_reference
            )

            concept_score, concept_details = compute_concept_score(
                student_answer, rubric, total_weight
            )
            gated_concept_score = concept_score * off_topic_gate

            structural_score = compute_structural_score(student_answer)

            n_c = len(rubric)
            adj_sims = [float(d["adjusted_similarity"]) for d in concept_details]
            cov_ratio = float(soft_coverage_ratio(adj_sims)) if adj_sims else 0.0
            contra_ratio = compute_contradiction_ratio(concept_details, n_c)

            hybrid_score = compute_hybrid_score(
                concept_score=gated_concept_score,
                structural_score=structural_score,
                semantic_score=semantic_score,
                coverage_ratio=cov_ratio,
                contradiction_ratio=contra_ratio,
                off_topic_gate=float(off_topic_gate),
            )

            raw_score_10 = float(hybrid_score) / 10.0

            human_scores.append(float(human_score))
            raw_engine_scores.append(raw_score_10)

    human_arr = np.array(human_scores, dtype=float)
    raw_arr = np.array(raw_engine_scores, dtype=float)

    calibrated = calibrate_scores(
        raw_scores=raw_arr,
        human_scores=human_arr,
        target_min=1.0,
        target_max=10.0,
    )

    pearson_corr, _ = pearsonr(human_arr, calibrated)
    spearman_corr, _ = spearmanr(human_arr, calibrated)
    mae_raw = float(np.mean(np.abs(human_arr - raw_arr)))
    mae_cal = float(np.mean(np.abs(human_arr - calibrated)))

    print(f"Pearson  Correlation : {pearson_corr:.3f}")
    print(f"Spearman Correlation : {spearman_corr:.3f}")
    print(f"MAE (raw)            : {mae_raw:.3f}")
    print(f"MAE (calibrated)     : {mae_cal:.3f}")
    print(f"Score range (raw)    : {raw_arr.max() - raw_arr.min():.2f}")
    print(f"Score range (calib)  : {calibrated.max() - calibrated.min():.2f}")


if __name__ == "__main__":
    main()

