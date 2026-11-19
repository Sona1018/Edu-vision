"""
concept_detection.py
--------------------
Core concept scoring logic for IntelliPlace.
Uses SBERT + DeBERTa NLI with softened hinge and relaxed contradiction logic.
"""

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from engine.embedding import get_embedding, get_entailment_score

concept_embedding_cache: dict = {}

TOP_K_SENTENCES = 3
TOP_K_WEIGHTS   = [1.0, 0.50, 0.20]
DOC_LEVEL_BLEND = 0.20

HINGE_LOW  = 0.26
HINGE_MID  = 0.50
HINGE_HIGH = 0.72

CONTRADICTION_GATE_THRESH = 0.72
ENTAILMENT_SHRINK_THRESH  = 0.66

# When SBERT blended similarity is already above this, skip the entailment-based shrink.
# Casual student prose often reads as "neutral" vs formal rubric text even when clearly on-topic.
SBERT_SKIP_ENTAILMENT_SHRINK_IF_BLEND_GE = 0.52

CONCEPT_COVERAGE_BONUS = 0.03
COVERAGE_THRESHOLD     = 0.31

# Penalize reusing the same sentence as primary evidence for multiple concepts (SBERT argmax collision).
PRIMARY_SENTENCE_REUSE_PENALTY = 0.52

def _hinge_remap(raw_sim: float) -> float:
    if raw_sim < HINGE_LOW:
        return 0.0
    elif raw_sim < HINGE_MID:
        return ((raw_sim - HINGE_LOW) / (HINGE_MID - HINGE_LOW)) * 0.40
    elif raw_sim < HINGE_HIGH:
        return 0.40 + ((raw_sim - HINGE_MID) / (HINGE_HIGH - HINGE_MID)) * 0.40
    else:
        return 0.80 + ((raw_sim - HINGE_HIGH) / (1.0 - HINGE_HIGH)) * 0.20

def split_sentences(text: str) -> list:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    clean = [s.strip() for s in sentences if len(s.strip()) > 8]
    return clean if clean else ([text.strip()] if text.strip() else [])

def compute_concept_score(student_answer: str, rubric: list, total_weight: float = 1.0):
    sentences = split_sentences(student_answer)

    if len(sentences) == 0:
        return 0.0, []

    sentence_embeddings = np.array([get_embedding(s) for s in sentences])
    full_answer_emb     = get_embedding(student_answer)

    weighted_score           = 0.0
    concept_details          = []
    concepts_above_threshold = 0
    used_primary_indices: set[int] = set()

    n_sent = len(sentences)
    n_con = len(rubric)
    slack = max(0.0, float(n_con + 2 - n_sent))
    # Few sentences vs many rubric items → soften reuse penalty so one good paragraph isn't fragmented.
    reuse_penalty_mult = PRIMARY_SENTENCE_REUSE_PENALTY + (1.0 - PRIMARY_SENTENCE_REUSE_PENALTY) * min(
        1.0, slack / 3.0
    )

    for concept in rubric:
        reference = concept["reference_explanation"]
        weight    = concept["weight"]

        if reference not in concept_embedding_cache:
            concept_embedding_cache[reference] = get_embedding(reference)
        ref_embedding = concept_embedding_cache[reference]

        # Sentence-level similarities
        sims_ref = cosine_similarity(sentence_embeddings, [ref_embedding]).flatten()
        raw_similarities = np.clip(sims_ref, 0.0, 1.0)
        sim_std = float(np.std(raw_similarities))

        # Prefer distinct supporting sentences across concepts (reduces false "missed" on later rubric items).
        pick_scores = raw_similarities.astype(float).copy()
        for idx in used_primary_indices:
            pick_scores[idx] *= reuse_penalty_mult
        best_idx = int(np.argmax(pick_scores))
        max_raw_sim = float(raw_similarities[best_idx])
        best_sentence = sentences[best_idx]
        used_primary_indices.add(best_idx)

        # Top-K blending
        k = min(TOP_K_SENTENCES, len(sentences))
        top_indices = np.argsort(raw_similarities)[-k:][::-1]
        
        weights_used = TOP_K_WEIGHTS[:k]
        weights_used = [w / sum(weights_used) for w in weights_used]
        
        top_k_quality = 0.0
        for rank, idx in enumerate(top_indices):
            top_k_quality += weights_used[rank] * float(raw_similarities[idx])

        # Document-level context
        doc_sim = float(cosine_similarity([full_answer_emb], [ref_embedding])[0][0])
        doc_sim = float(np.clip(doc_sim, 0.0, 1.0))

        blended_sim = (1.0 - DOC_LEVEL_BLEND) * top_k_quality + DOC_LEVEL_BLEND * doc_sim
        hinge_sim   = _hinge_remap(blended_sim)

        # NLI (Relaxed)
        entailment_prob, contradiction_prob = get_entailment_score(best_sentence, reference)
        
        adjusted = hinge_sim
        ent = entailment_prob
        con = contradiction_prob

        if con > CONTRADICTION_GATE_THRESH and con > ent:
            # Document-aligned answers still trigger spurious NLI contradiction vs formal rubric text.
            if doc_sim >= 0.50:
                adjusted *= 0.66
            else:
                adjusted *= 0.52
        elif con > 0.52 and con > ent + 0.08:
            adjusted *= 0.72

        if adjusted > 0.0 and ent < ENTAILMENT_SHRINK_THRESH:
            if blended_sim < SBERT_SKIP_ENTAILMENT_SHRINK_IF_BLEND_GE:
                adjusted *= 0.82

        adjusted = float(np.clip(adjusted, 0.0, 1.0))

        if adjusted >= COVERAGE_THRESHOLD:
            concepts_above_threshold += 1

        concept_contribution = weight * adjusted
        weighted_score      += concept_contribution

        concept_details.append({
            "concept_id":             concept["concept_id"],
            "concept_name":           concept["concept_name"],
            "best_sentence":          best_sentence,
            "raw_sbert_sim":          round(max_raw_sim, 4),
            "combined_quality":       round(blended_sim, 4),
            "adjusted_similarity":    round(adjusted, 4),
            "weighted_score":         round(concept_contribution, 4),
            "sim_std":                round(sim_std, 4),
            "contradiction":          round(contradiction_prob, 4),
            "entailment":             round(entailment_prob, 4),
            "depth_score":            0.0
        })

    base_score = (weighted_score / total_weight) * 100.0

    total_concepts = len(rubric)
    if total_concepts > 0:
        coverage_ratio = concepts_above_threshold / total_concepts
        coverage_bonus = CONCEPT_COVERAGE_BONUS * coverage_ratio * 100.0
    else:
        coverage_bonus = 0.0

    concept_score = float(np.clip(base_score + coverage_bonus, 0.0, 100.0))

    return concept_score, concept_details