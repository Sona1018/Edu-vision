"""evaluate_engine.py — dataset loop, hybrid on 0–100, then calibration to 1–10."""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr

from engine.mastery_vector import MasteryVector
from engine.concept_analysis import analyze_mastery
from engine.scoring import compute_semantic_score
from engine.concept_detection import compute_concept_score
from engine.mastery import (
    compute_contradiction_ratio,
    compute_hybrid_score,
    compute_structural_score,
    soft_coverage_ratio,
)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_scores(
    raw_scores:   np.ndarray,
    human_scores: np.ndarray = None,
    target_min:   float = 10.0,    # FIX: calibrate on 0–100 scale, not 0–10
    target_max:   float = 100.0,
) -> np.ndarray:
    """
    Two-step post-hoc calibration operating on the FULL [0,100] scale.

    FIX v7: Previously raw_scores were already divided by 10 before calibration,
    compressing the input range to [0,10]. This meant calibration was stretching
    a signal that had already lost resolution. Now:
      - calibrate_scores receives raw hybrid scores in [0, 100]
      - stretches to [10, 100] (1/10 equivalent of [1, 10])
      - bias correction applied in [0, 100] space
      - final conversion to [1, 10] happens AFTER calibration

    Step 1 — Linear stretch: [raw.min, raw.max] → [target_min, target_max]
    Step 2 — Bias correction: shift mean to match human mean (in [1,10] space)
    """
    raw   = np.array(raw_scores, dtype=float)
    r_min = raw.min()
    r_max = raw.max()

    if r_max == r_min:
        return np.full_like(raw, (target_min + target_max) / 2.0)

    # Step 1: stretch to [10, 100]
    stretched = (raw - r_min) / (r_max - r_min)
    stretched = stretched * (target_max - target_min) + target_min

    # Convert to [1, 10] for bias correction against human scores
    stretched_10 = stretched / 10.0

    # Step 2: bias correction in [1, 10] space
    if human_scores is not None:
        human_mean  = np.array(human_scores, dtype=float).mean()
        engine_mean = stretched_10.mean()
        bias        = human_mean - engine_mean
        stretched_10 = np.clip(stretched_10 + bias, 1.0, 10.0)

    return stretched_10


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

print("=" * 60)
print("IntelliPlace Evaluation Engine")
print("=" * 60)

mastery_system = MasteryVector(alpha=0.4)

with open("dataset/intelliplace_dataset.json", "r") as f:
    dataset = json.load(f)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

human_scores      = []
raw_engine_scores = []   # pre-calibration, in [0, 100] now (NOT /10)

total_answers = sum(len(q["answers"]) for q in dataset)
print(f"\nEvaluating {total_answers} answers...\n")

counter = 1

for q in dataset:

    rubric       = q["rubric"]
    total_weight = sum(c["weight"] for c in rubric)

    combined_reference = " ".join(
        c["reference_explanation"] for c in rubric
    )

    for ans in q["answers"]:

        student_answer = ans["text"]
        human_score    = ans["score"]

        # ── 1. Semantic score ──────────────────────────────────────────────
        semantic_score, off_topic_gate = compute_semantic_score(
            student_answer, combined_reference
        )

        # ── 2. Concept score ───────────────────────────────────────────────
        concept_score, concept_details = compute_concept_score(
            student_answer, rubric, total_weight
        )

        gated_concept_score = concept_score * off_topic_gate

        # ── 3. Structural score ────────────────────────────────────────────
        structural_score = compute_structural_score(student_answer)

        # ── 4. Coverage ratio ─────────────────────────────────────────────
        adj_sims = [float(d["adjusted_similarity"]) for d in concept_details]
        n_c      = len(rubric)
        if n_c > 0 and adj_sims:
            cov_ratio = float(soft_coverage_ratio(adj_sims))
        else:
            cov_ratio = 0.0

        # ── 5. Contradiction ratio (shared with meta-model feature pipeline)
        contradiction_ratio = compute_contradiction_ratio(concept_details, n_c)

        # ── 6. Hybrid score (0–100) — semantic + contradiction active in mastery
        hybrid_score = compute_hybrid_score(
            concept_score       = gated_concept_score,
            structural_score    = structural_score,
            semantic_score      = semantic_score,
            coverage_ratio      = cov_ratio,
            contradiction_ratio = contradiction_ratio,
            off_topic_gate      = float(off_topic_gate),
        )

        # FIX: Do NOT divide by 10 here. Keep on 0–100 for calibration.
        # Division to [0, 10] happens inside calibrate_scores after stretching.

        # ── 7. Mastery update ─────────────────────────────────────────────
        mastery_system.update({
            c["concept_id"]: c["adjusted_similarity"]
            for c in concept_details
        })

        # ── 8. Logging ────────────────────────────────────────────────────
        print(
            f"[{counter:3d}] | Human:{human_score:2d} | "
            f"Semantic:{semantic_score:5.1f} Gate:{off_topic_gate:.2f} | "
            f"Concept:{concept_score:5.1f} Gated:{gated_concept_score:5.1f} | "
            f"Struct:{structural_score:5.1f} | "
            f"Hybrid:{hybrid_score:5.1f} | "
            f"CovRatio:{cov_ratio:.2f} | ContraRatio:{contradiction_ratio:.2f}"
        )

        human_scores.append(human_score)
        raw_engine_scores.append(hybrid_score)   # store 0–100 value

        counter += 1


# ---------------------------------------------------------------------------
# Post-hoc calibration
# ---------------------------------------------------------------------------

human_arr  = np.array(human_scores,      dtype=float)
raw_arr    = np.array(raw_engine_scores, dtype=float)  # 0–100 scale

# calibrate_scores returns values in [1, 10]
calibrated = calibrate_scores(
    raw_scores   = raw_arr,
    human_scores = human_arr,
    target_min   = 10.0,
    target_max   = 100.0,
)

# Convert raw to [1, 10] for MAE comparison (just for display)
raw_arr_10 = np.clip(raw_arr / 10.0, 1.0, 10.0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

pearson_corr,  _ = pearsonr(human_arr,  calibrated)
spearman_corr, _ = spearmanr(human_arr, calibrated)
mae_raw          = float(np.mean(np.abs(human_arr - raw_arr_10)))
mae_calibrated   = float(np.mean(np.abs(human_arr - calibrated)))

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"\nHuman Scores       : {human_arr}")
print(f"Raw Eng (0-10)     : {np.round(raw_arr_10, 2)}")
print(f"Calibrated (1-10)  : {np.round(calibrated, 2)}")
print(f"\nPearson  Correlation : {pearson_corr:.3f}  (target >= 0.80)")
print(f"Spearman Correlation : {spearman_corr:.3f}  (target >= 0.80)")
print(f"MAE (raw)            : {mae_raw:.3f}")
print(f"MAE (calibrated)     : {mae_calibrated:.3f}  (target <= 1.5)")

score_range_raw   = raw_arr_10.max() - raw_arr_10.min()
score_range_calib = calibrated.max() - calibrated.min()
print(f"\nScore range (raw)    : {score_range_raw:.2f}  (target >= 6.0)")
print(f"Score range (calib)  : {score_range_calib:.2f}")


# ---------------------------------------------------------------------------
# Mastery vector
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STUDENT MASTERY VECTOR")
print("=" * 60)

mastery_vector = mastery_system.get_mastery()

for concept_id, m_score in mastery_vector.items():
    print(f"  {concept_id:30s} : {m_score:.3f}")

weak, moderate, strong = analyze_mastery(mastery_vector)

print("\n--- WEAK CONCEPTS (< 0.40) ---")
for c, s in weak:
    print(f"  {c} : {s:.3f}")

print("\n--- MODERATE CONCEPTS (0.40 – 0.70) ---")
for c, s in moderate:
    print(f"  {c} : {s:.3f}")

print("\n--- STRONG CONCEPTS (> 0.70) ---")
for c, s in strong:
    print(f"  {c} : {s:.3f}")

print("\n" + "=" * 60)
print("Evaluation complete.")
print("=" * 60)