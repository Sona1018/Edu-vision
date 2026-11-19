import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from engine.concept_detection import (
    COVERAGE_THRESHOLD,
    concept_embedding_cache,
    compute_concept_score,
    split_sentences,
)
from engine.embedding import get_embedding, get_entailment_score
from engine.mastery import (
    CONTRADICTION_RATIO_THRESH,
    compute_contradiction_ratio,
    compute_hybrid_score,
    compute_structural_score,
    soft_coverage_ratio,
)
from engine.scoring import compute_semantic_score

# Passive NLI feature only (not used in engine / hybrid). Chunk = same split as concept_detection.
CONTRADICTION_CHUNK_PROB_THRESH = 0.7


def _passive_contradiction_score(answer_text: str, rubric: list) -> float:
    """
    Fraction of answer chunks whose NLI contradiction prob vs the best-matching
    rubric reference (by SBERT cosine) exceeds CONTRADICTION_CHUNK_PROB_THRESH.

    Each chunk is aligned to exactly one reference (argmax similarity), then one
    NLI forward pass per chunk — no chunk×concept NLI grid.
    """
    sentences = split_sentences(answer_text)
    if not sentences:
        return 0.0
    ref_texts = [
        str(c.get("reference_explanation", "")).strip()
        for c in rubric
        if str(c.get("reference_explanation", "")).strip()
    ]
    if not ref_texts:
        return 0.0

    for ref in ref_texts:
        if ref not in concept_embedding_cache:
            concept_embedding_cache[ref] = get_embedding(ref)
    ref_mat = np.array([concept_embedding_cache[r] for r in ref_texts], dtype=float)

    sentence_embeddings = np.array([get_embedding(s) for s in sentences], dtype=float)
    sims = cosine_similarity(sentence_embeddings, ref_mat)
    n_high = 0
    for i, sentence in enumerate(sentences):
        j = int(np.argmax(sims[i]))
        reference = ref_texts[j]
        _ent, con = get_entailment_score(sentence, reference)
        c = float(con)
        if not np.isfinite(c):
            c = 0.0
        if c > CONTRADICTION_CHUNK_PROB_THRESH:
            n_high += 1
    return float(n_high) / float(len(sentences))


def _safe_stats(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.max()), float(arr.min()), float(arr.std())


def _print_feature_sanity(df: pd.DataFrame) -> None:
    """Print compact sanity stats for key numeric features."""
    key_features = [
        "semantic_score",
        "structural_score",
        "concept_score",
        "concept_max",
        "concept_min",
        "concept_std",
        "coverage_ratio",
        "answer_length",
        "concept_count_matched",
        "consistency_score",
        "weak_signal",
        "concept_range",
        "dominance_score",
        "consistency_ratio",
        "contradiction_score",
        "contradiction_severity",
    ]
    print("\nFeature sanity (variance check):")
    for col in key_features:
        if col not in df.columns:
            print(f"  - {col}: MISSING")
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        non_null = int(series.notna().sum())
        unique = int(series.nunique(dropna=True))
        std = float(series.std(ddof=0)) if non_null > 0 else 0.0
        mn = float(series.min()) if non_null > 0 else 0.0
        mx = float(series.max()) if non_null > 0 else 0.0
        flag = " [near-constant]" if unique <= 1 or std < 1e-6 else ""
        print(
            f"  - {col}: std={std:.4f}, min={mn:.4f}, max={mx:.4f}, "
            f"unique={unique}, non_null={non_null}{flag}"
        )


def build_rows(dataset: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for q in dataset:
        question_id = q.get("question_id", "unknown")
        rubric = q["rubric"]
        n_concepts = len(rubric)
        total_weight = float(sum(c["weight"] for c in rubric))
        combined_reference = " ".join(c["reference_explanation"] for c in rubric)

        for ans in q["answers"]:
            answer_text = ans["text"]
            human_score = float(ans["score"])
            answer_words = answer_text.split()

            semantic_score, off_topic_gate = compute_semantic_score(answer_text, combined_reference)
            concept_score, concept_details = compute_concept_score(answer_text, rubric, total_weight)
            gated_concept_score = concept_score * off_topic_gate
            structural_score = compute_structural_score(answer_text)
            # Passive NLI: chunk-level contradiction rate (meta-model only)
            contradiction_score = _passive_contradiction_score(answer_text, rubric)

            # Compute coverage_ratio for hybrid constraints
            adj_for_cov = [float(d["adjusted_similarity"]) for d in concept_details]
            if n_concepts > 0 and adj_for_cov:
                coverage_ratio_val = float(soft_coverage_ratio(adj_for_cov))
            else:
                coverage_ratio_val = 0.0

            contradiction_severity = float(contradiction_score) * (1.0 - coverage_ratio_val)

            contradiction_ratio_val = compute_contradiction_ratio(
                concept_details, n_concepts
            )

            hybrid_score = compute_hybrid_score(
                concept_score=gated_concept_score,
                structural_score=structural_score,
                semantic_score=semantic_score,
                coverage_ratio=coverage_ratio_val,
                contradiction_ratio=contradiction_ratio_val,
                off_topic_gate=float(off_topic_gate),
            )
            raw_score_10 = float(hybrid_score) / 10.0

            adjusted = [float(d["adjusted_similarity"]) for d in concept_details]
            contradiction = [float(d["contradiction"]) for d in concept_details]
            entailment = [float(d["entailment"]) for d in concept_details]
            depth = [float(d["depth_score"]) for d in concept_details]
            combined = [float(d["combined_quality"]) for d in concept_details]
            # v9: per-concept sentence similarity variance
            sim_stds = [float(d.get("sim_std", 0.0)) for d in concept_details]
            # v9: raw SBERT similarity per concept (before hinge)
            raw_sims = [float(d.get("raw_sbert_sim", 0.0)) for d in concept_details]

            adj_mean, adj_max, adj_min, adj_std = _safe_stats(adjusted)
            ent_mean, ent_max, ent_min, _ = _safe_stats(entailment)
            con_mean, con_max, _, _ = _safe_stats(contradiction)
            dep_mean, dep_max, _, _ = _safe_stats(depth)
            cmb_mean, cmb_max, cmb_min, cmb_std = _safe_stats(combined)
            sim_std_mean, sim_std_max, _, _ = _safe_stats(sim_stds)
            raw_sim_mean, raw_sim_max, raw_sim_min, raw_sim_std = _safe_stats(raw_sims)

            coverage_04 = float(sum(v >= 0.40 for v in adjusted)) / max(1, n_concepts)
            coverage_06 = float(sum(v >= 0.60 for v in adjusted)) / max(1, n_concepts)
            coverage_03 = float(sum(v >= 0.30 for v in adjusted)) / max(1, n_concepts)
            contradiction_count = float(
                sum(v >= CONTRADICTION_RATIO_THRESH for v in contradiction)
            )

            # v9: zero-credit concept count — how many concepts got hard-cutoff
            zero_credit_count = float(sum(v == 0.0 for v in adjusted))
            zero_credit_ratio = zero_credit_count / max(1, n_concepts)

            # v9: concept score spread — high spread = uneven coverage
            adj_range = adj_max - adj_min if adjusted else 0.0

            # Derived from same adj_* and coverage_ratio_val (no extra passes)
            consistency_score = float(adj_max - adj_std)
            weak_signal = float(adj_min * coverage_ratio_val)
            concept_range = float(adj_max - adj_min)
            dominance_score = float(adj_max * coverage_ratio_val)
            consistency_ratio = float(adj_min / (adj_max + 1e-6))

            rows.append(
                {
                    "question_id": question_id,
                    "answer_id": ans.get("answer_id", ""),
                    "human_score": human_score,
                    "answer_len_words": float(len(answer_words)),
                    "answer_len_chars": float(len(answer_text)),
                    "n_concepts": float(n_concepts),
                    "total_weight": total_weight,
                    "semantic_score": float(semantic_score),
                    "off_topic_gate": float(off_topic_gate),
                    "concept_score": float(concept_score),
                    "gated_concept_score": float(gated_concept_score),
                    "structural_score": float(structural_score),
                    "hybrid_score_100": float(hybrid_score),
                    "raw_score_10": raw_score_10,
                    # Requested stable aliases for meta-model separability
                    "answer_length": float(len(answer_words)),
                    "coverage_ratio": float(coverage_ratio_val),
                    "concept_max": adj_max,
                    "concept_min": adj_min,
                    "concept_std": adj_std,
                    "concept_count_matched": float(
                        sum(v >= COVERAGE_THRESHOLD for v in adjusted)
                    ),
                    "consistency_score": consistency_score,
                    "weak_signal": weak_signal,
                    "concept_range": concept_range,
                    "dominance_score": dominance_score,
                    "consistency_ratio": consistency_ratio,
                    "contradiction_score": float(contradiction_score),
                    "contradiction_severity": contradiction_severity,
                    "adj_mean": adj_mean,
                    "adj_max": adj_max,
                    "adj_min": adj_min,
                    "adj_std": adj_std,
                    "adj_range": adj_range,
                    "coverage_03": coverage_03,
                    "coverage_04": coverage_04,
                    "coverage_06": coverage_06,
                    "zero_credit_count": zero_credit_count,
                    "zero_credit_ratio": zero_credit_ratio,
                    "contradiction_count": contradiction_count,
                    "contradiction_mean": con_mean,
                    "contradiction_max": con_max,
                    "entailment_mean": ent_mean,
                    "entailment_max": ent_max,
                    "entailment_min": ent_min,
                    "depth_mean": dep_mean,
                    "depth_max": dep_max,
                    "combined_mean": cmb_mean,
                    "combined_max": cmb_max,
                    "combined_min": cmb_min,
                    "combined_std": cmb_std,
                    # v9: new variance/spread features
                    "sim_std_mean": sim_std_mean,
                    "sim_std_max": sim_std_max,
                    "raw_sim_mean": raw_sim_mean,
                    "raw_sim_max": raw_sim_max,
                    "raw_sim_min": raw_sim_min,
                    "raw_sim_std": raw_sim_std,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature table for meta-model training.")
    parser.add_argument("--dataset", default="dataset/intelliplace_dataset.json")
    parser.add_argument("--output", default="dataset/meta_features.csv")
    args, _unknown = parser.parse_known_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    rows = build_rows(dataset)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    # Ensure numeric consistency for core model signals.
    for col in (
        "semantic_score",
        "structural_score",
        "concept_score",
        "contradiction_score",
        "contradiction_severity",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
    _print_feature_sanity(df)
    try:
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df)} rows to {out_path}")
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = out_path.with_name(f"{out_path.stem}_{ts}{out_path.suffix}")
        df.to_csv(fallback, index=False)
        print(f"Target file was locked: {out_path}. Wrote {len(df)} rows to fallback path: {fallback}")


if __name__ == "__main__":
    main()
