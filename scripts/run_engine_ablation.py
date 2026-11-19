import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

import engine.concept_detection as concept_detection
import engine.mastery as mastery
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


def evaluate_dataset(dataset: list[dict], max_rows_per_question: int | None = None) -> dict:
    human_scores = []
    raw_scores = []
    for q in dataset:
        rubric = q["rubric"]
        total_weight = float(sum(c["weight"] for c in rubric))
        combined_reference = " ".join(c["reference_explanation"] for c in rubric)
        answers = q["answers"]
        if max_rows_per_question is not None:
            answers = answers[:max_rows_per_question]
        for ans in answers:
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
    concept_detection.CONCEPT_COVERAGE_BONUS = cfg["coverage_bonus"]
    wc = float(cfg["weight_concept"])
    ws = float(cfg["weight_struct"])
    wsem = float(cfg.get("weight_semantic", max(0.0, round(1.0 - wc - ws, 4))))
    mastery.WEIGHT_CONCEPT = wc
    mastery.WEIGHT_STRUCT = ws
    mastery.WEIGHT_SEMANTIC = wsem


def config_space() -> list[dict]:
    return [
        {
            "name": "baseline_reference",
            "contra_thresh": 0.58,
            "contra_factor": 0.50,
            "nli_top_k": 1,
            "coverage_thresh": 0.40,
            "coverage_bonus": 0.04,
            "weight_concept": 0.88,
            "weight_struct": 0.12,
        },
        {
            "name": "topk_conservative_gate",
            "contra_thresh": 0.72,
            "contra_factor": 0.75,
            "nli_top_k": 2,
            "coverage_thresh": 0.42,
            "coverage_bonus": 0.03,
            "weight_concept": 0.90,
            "weight_struct": 0.10,
        },
        {
            "name": "topk_medium_gate",
            "contra_thresh": 0.68,
            "contra_factor": 0.75,
            "nli_top_k": 2,
            "coverage_thresh": 0.42,
            "coverage_bonus": 0.03,
            "weight_concept": 0.90,
            "weight_struct": 0.10,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run focused engine ablation experiments.")
    parser.add_argument("--dataset", default="dataset/intelliplace_dataset.json")
    parser.add_argument("--report-out", default="artifacts/engine_ablation_report.json")
    parser.add_argument(
        "--max-rows-per-question",
        type=int,
        default=1,
        help="Cap rows per question for faster ablation iteration. Use 0 to disable cap.",
    )
    args, _unknown = parser.parse_known_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    max_rows = args.max_rows_per_question if args.max_rows_per_question > 0 else None
    results = []
    for cfg in config_space():
        apply_config(cfg)
        metrics = evaluate_dataset(dataset, max_rows_per_question=max_rows)
        objective = metrics["pearson"] + 0.5 * metrics["spearman"] - 0.03 * metrics["mae_cal"]
        row = {"objective": float(objective), "config": cfg, "metrics": metrics}
        results.append(row)
        print(
            f"{cfg['name']}: pearson={metrics['pearson']:.3f} "
            f"spearman={metrics['spearman']:.3f} mae_cal={metrics['mae_cal']:.3f}"
        )

    results_sorted = sorted(results, key=lambda r: r["objective"], reverse=True)
    report = {
        "rows_evaluated": int(
            sum(
                min(len(q["answers"]), max_rows) if max_rows is not None else len(q["answers"])
                for q in dataset
            )
        ),
        "max_rows_per_question": max_rows,
        "results": results_sorted,
        "best": results_sorted[0],
    }
    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nBest config: {results_sorted[0]['config']['name']}")
    print(f"Wrote ablation report to: {out}")


if __name__ == "__main__":
    main()
