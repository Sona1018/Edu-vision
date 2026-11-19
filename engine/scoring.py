"""
scoring.py
----------
Semantic similarity (0–100) for features and a mild off-topic gate on concept.
Linear baseline correction only — no gamma shaping (preserves variance for meta-model).
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from engine.embedding import get_embedding

OFF_TOPIC_THRESHOLD = 0.35
GATE_MIN            = 0.48
GATE_MAX            = 1.00
SIMILARITY_BASELINE = 0.12


def compute_semantic_score(student_answer: str, ideal_answer: str):
    """
    Baseline-corrected cosine similarity in [0, 100], plus off-topic gate.

    Returns
    -------
    score : float in [0, 100]
        Linear map from corrected cosine; stable feature for meta-model.

    off_topic_gate : float in [GATE_MIN, GATE_MAX]
        Scales concept when raw similarity is below OFF_TOPIC_THRESHOLD.
    """
    student_emb = get_embedding(student_answer)
    ideal_emb   = get_embedding(ideal_answer)

    raw_similarity = float(
        cosine_similarity([student_emb], [ideal_emb])[0][0]
    )
    raw_similarity = max(0.0, raw_similarity)

    corrected = (raw_similarity - SIMILARITY_BASELINE) / (1.0 - SIMILARITY_BASELINE)
    corrected = float(np.clip(corrected, 0.0, 1.0))

    score = round(corrected * 100.0, 2)

    # Gate uses a blend so baseline-corrected similarity can pull borderline answers below 1.0
    # without changing the reported semantic_score (still linear corrected × 100).
    gate_signal = 0.58 * raw_similarity + 0.42 * corrected
    if gate_signal < OFF_TOPIC_THRESHOLD:
        gate_ratio     = gate_signal / OFF_TOPIC_THRESHOLD
        off_topic_gate = GATE_MIN + gate_ratio * (GATE_MAX - GATE_MIN)
    else:
        off_topic_gate = GATE_MAX

    off_topic_gate = float(np.clip(off_topic_gate, GATE_MIN, GATE_MAX))

    return score, off_topic_gate
