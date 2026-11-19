import json
import numpy as np
from scipy.stats import pearsonr, spearmanr

import engine.concept_detection as concept_detection
import engine.mastery as mastery
import engine.scoring as scoring

from engine.concept_detection import COVERAGE_THRESHOLD, compute_concept_score
from engine.mastery import (
    compute_contradiction_ratio,
    compute_hybrid_score,
    compute_structural_score,
    soft_coverage_ratio,
)
from engine.scoring import compute_semantic_score


def calibrate_scores(raw_scores: np.ndarray, human_scores: np.ndarray) -> np.ndarray:
    r_min = raw_scores.min()
    r_max = raw_scores.max()
    if r_max == r_min:
        return np.full_like(raw_scores, 5.5)
    stretched = (raw_scores - r_min) / (r_max - r_min)
    stretched = stretched * 9.0 + 1.0
    bias = float(np.mean(human_scores) - np.mean(stretched))
    return np.clip(stretched + bias, 1.0, 10.0)


def evaluate_dataset(dataset: list) -> dict:
    human_scores = []
    raw_scores = []
    for q in dataset:
        rubric = q["rubric"]
        total_weight = float(sum(c["weight"] for c in rubric))
        combined_reference = " ".join(c["reference_explanation"] for c in rubric)
        for ans in q["answers"]:
            text = ans["text"]
            human = float(ans["score"])
            sem, gate = compute_semantic_score(text, combined_reference)
            concept_score, details = compute_concept_score(text, rubric, total_weight)
            concept_score *= gate
            structural = compute_structural_score(text)
            n_c = len(rubric)
            adj = [float(d["adjusted_similarity"]) for d in details]
            cov = float(soft_coverage_ratio(adj)) if adj else 0.0
            cr = compute_contradiction_ratio(details, n_c)
            hybrid = compute_hybrid_score(
                concept_score=concept_score,
                structural_score=structural,
                semantic_score=sem,
                coverage_ratio=cov,
                contradiction_ratio=cr,
                off_topic_gate=float(gate),
            )
            raw10 = float(hybrid) / 10.0
            human_scores.append(human)
            raw_scores.append(raw10)

    h = np.array(human_scores, dtype=float)
    r = np.array(raw_scores, dtype=float)
    c = calibrate_scores(r, h)
    p, _ = pearsonr(h, c)
    s, _ = spearmanr(h, c)
    mae_raw = float(np.mean(np.abs(h - r)))
    mae_cal = float(np.mean(np.abs(h - c)))
    return {
        "pearson": float(p),
        "spearman": float(s),
        "mae_raw": mae_raw,
        "mae_cal": mae_cal,
        "range_raw": float(r.max() - r.min()),
        "range_cal": float(c.max() - c.min()),
    }


def apply_config(cfg: dict) -> None:
    concept_detection.CONTRADICTION_GATE_THRESH = cfg["contra_thresh"]
    concept_detection.COVERAGE_THRESHOLD = cfg["coverage_thresh"]
    scoring.OFF_TOPIC_THRESHOLD = cfg["off_topic_thresh"]
    scoring.GATE_MIN = cfg["gate_min"]
    wc = float(cfg["weight_concept"])
    ws = float(cfg["weight_struct"])
    mastery.WEIGHT_CONCEPT = wc
    mastery.WEIGHT_STRUCT = ws
    mastery.WEIGHT_SEMANTIC = float(cfg.get("weight_semantic", max(0.0, round(1.0 - wc - ws, 4))))


def config_space():
    # Focused set around known-good region to keep turnaround practical.
    # (Full Cartesian sweep is too slow with transformer inference.)
    return [
        {"name_blend": 0.22, "contra_thresh": 0.58, "depth_scale": 0.06, "coverage_thresh": 0.40, "off_topic_thresh": 0.35, "gate_min": 0.50, "weight_concept": 0.88, "weight_struct": 0.12, "top_recovery": 0.85},
        {"name_blend": 0.24, "contra_thresh": 0.58, "depth_scale": 0.06, "coverage_thresh": 0.40, "off_topic_thresh": 0.35, "gate_min": 0.50, "weight_concept": 0.88, "weight_struct": 0.12, "top_recovery": 0.85},
        {"name_blend": 0.26, "contra_thresh": 0.60, "depth_scale": 0.06, "coverage_thresh": 0.40, "off_topic_thresh": 0.35, "gate_min": 0.50, "weight_concept": 0.88, "weight_struct": 0.12, "top_recovery": 0.85},
        {"name_blend": 0.24, "contra_thresh": 0.60, "depth_scale": 0.04, "coverage_thresh": 0.42, "off_topic_thresh": 0.35, "gate_min": 0.50, "weight_concept": 0.88, "weight_struct": 0.12, "top_recovery": 0.85},
        {"name_blend": 0.24, "contra_thresh": 0.58, "depth_scale": 0.04, "coverage_thresh": 0.40, "off_topic_thresh": 0.33, "gate_min": 0.50, "weight_concept": 0.88, "weight_struct": 0.12, "top_recovery": 0.85},
        {"name_blend": 0.24, "contra_thresh": 0.58, "depth_scale": 0.06, "coverage_thresh": 0.40, "off_topic_thresh": 0.35, "gate_min": 0.45, "weight_concept": 0.88, "weight_struct": 0.12, "top_recovery": 0.85},
        {"name_blend": 0.24, "contra_thresh": 0.58, "depth_scale": 0.06, "coverage_thresh": 0.40, "off_topic_thresh": 0.35, "gate_min": 0.50, "weight_concept": 0.90, "weight_struct": 0.10, "top_recovery": 0.82},
        {"name_blend": 0.24, "contra_thresh": 0.58, "depth_scale": 0.06, "coverage_thresh": 0.40, "off_topic_thresh": 0.35, "gate_min": 0.50, "weight_concept": 0.90, "weight_struct": 0.10, "top_recovery": 0.88},
    ]


def main():
    with open("dataset/intelliplace_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    for i, cfg in enumerate(config_space(), start=1):
        apply_config(cfg)
        m = evaluate_dataset(dataset)
        # primary objective: calibrated correlation; secondary: calibrated MAE
        score = m["pearson"] + 0.5 * m["spearman"] - 0.03 * m["mae_cal"]
        row = {"idx": i, "objective": score, **cfg, **m}
        results.append(row)
        print(
            f"#{i:03d} p={m['pearson']:.3f} s={m['spearman']:.3f} "
            f"mae={m['mae_cal']:.3f} cfg={cfg}"
        , flush=True)

    best = sorted(results, key=lambda x: x["objective"], reverse=True)[:10]
    print("\n=== TOP 10 CONFIGS ===")
    for b in best:
        print(json.dumps(b, indent=2))


if __name__ == "__main__":
    main()
