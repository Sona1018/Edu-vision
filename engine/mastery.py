"""
mastery.py
----------
Structural score and hybrid combination.

Hybrid is a stable blend for a raw signal / meta-model anchor; strong
ranking refinement is left to calibration and the meta-model.
"""

import numpy as np

# Concept-dominant; semantic/structure are secondary so strong rubric coverage moves hybrid up.
WEIGHT_CONCEPT   = 0.78
WEIGHT_STRUCT   = 0.09
WEIGHT_SEMANTIC  = 0.13

STRUCTURAL_CEILING = 78.0

# Soft ceilings — only curb extreme mismatch; binary coverage used to hit NONE too often.
COVERAGE_CAP_60   = 82.0
COVERAGE_CAP_40   = 70.0
COVERAGE_CAP_NONE = 50.0

# Partial credit band below COVERAGE_THRESHOLD for hybrid caps / meta coverage_ratio feature.
SOFT_COVERAGE_BAND = 0.11


def _gate_coherence_adjustments(
    concept_score: float,
    semantic_score: float,
    coverage_ratio: float | None,
    off_topic_gate: float | None,
) -> tuple[float, float]:
    """Reduce weak-answer inflation and strong-answer underestimation without new models.

    * Weak pattern: high holistic ``semantic_score`` but low ``concept_score`` and thin
      ``coverage_ratio`` → down-weight semantic (keyword overlap without rubric support).
    * Strong pattern: high ``off_topic_gate``, decent soft ``coverage_ratio``, but
      aggregated ``concept_score`` still depressed by hinge/NLI style mismatch → small uplift.
    """
    c = float(np.clip(concept_score, 0.0, 100.0))
    sem = float(np.clip(semantic_score, 0.0, 100.0))
    cov = coverage_ratio
    g = off_topic_gate

    # --- Weak: semantic free-lunch when rubric evidence is thin ---
    if cov is not None and c < 36.0 and sem > 54.0 and cov < 0.34:
        t = min(1.0, max(0.0, float(cov) / 0.34))
        sem *= 0.50 + 0.50 * t

    # --- Strong: clearly on-topic + breadth, but concept aggregate still low ---
    if (
        g is not None
        and float(g) >= 0.96
        and cov is not None
        and float(cov) >= 0.36
        and c < 52.0
        and sem >= 48.0
    ):
        room = max(0.0, 52.0 - c)
        cov_excess = max(0.0, float(cov) - 0.36) / 0.64
        boost = min(15.0, room * 0.35 + cov_excess * 11.0)
        c = min(76.0, c + boost)

    return float(np.clip(c, 0.0, 100.0)), float(np.clip(sem, 0.0, 100.0))


def soft_coverage_ratio(adjusted_similarities: list[float]) -> float:
    """Fraction of rubric concepts covered, with linear partial credit just below threshold."""
    if not adjusted_similarities:
        return 0.0
    from engine import concept_detection as cd  # runtime read — respects tuning scripts that patch threshold

    thresh = float(cd.COVERAGE_THRESHOLD)
    band = float(SOFT_COVERAGE_BAND)
    low = thresh - band
    parts: list[float] = []
    for v in adjusted_similarities:
        vf = float(v)
        if vf >= thresh:
            parts.append(1.0)
        elif vf >= low:
            parts.append(max(0.0, (vf - low) / band))
        else:
            parts.append(0.0)
    return sum(parts) / float(len(parts))

CONTRADICTION_RATIO_THRESH = 0.62
# Additive penalty on 0–100 hybrid (at contradiction_ratio=1, subtract this many points).
CONTRADICTION_ADDITIVE_MAX = 12.0


def compute_contradiction_ratio(concept_details: list, n_concepts: int) -> float:
    if not concept_details or n_concepts <= 0:
        return 0.0
    n = sum(
        1
        for d in concept_details
        if float(d.get("contradiction", 0.0)) >= CONTRADICTION_RATIO_THRESH
    )
    return n / float(n_concepts)


def compute_structural_score(student_answer: str) -> float:
    """Word-count rubric; ceiling at STRUCTURAL_CEILING."""
    words  = student_answer.split()
    length = len(words)

    if length < 10:
        score = (length / 10.0) * 14.0
    elif length < 30:
        score = 14.0 + ((length - 10) / 20.0) * 30.0
    elif length < 60:
        score = 44.0 + ((length - 30) / 30.0) * 24.0
    elif length < 100:
        score = 68.0 + ((length - 60) / 40.0) * 8.0
    else:
        score = STRUCTURAL_CEILING

    return round(float(np.clip(score, 0.0, STRUCTURAL_CEILING)), 2)


def compute_hybrid_score(
    concept_score: float,
    structural_score: float,
    semantic_score: float = 0.0,
    coverage_ratio: float = None,
    contradiction_ratio: float = 0.0,
    off_topic_gate: float | None = None,
) -> float:
    """
    Blend gated concept, holistic semantic (0–100), and length.
    Coverage soft-caps, then a small additive contradiction adjustment.

    ``off_topic_gate`` (optional, 0–1 from ``compute_semantic_score``) enables
    coherence shaping: modest concept uplift when clearly on-topic with breadth,
    and semantic down-weight when holistic similarity outruns rubric coverage.
    """
    c_raw = float(np.clip(concept_score, 0.0, 100.0))
    sem_raw = float(np.clip(semantic_score, 0.0, 100.0))
    c, sem = _gate_coherence_adjustments(c_raw, sem_raw, coverage_ratio, off_topic_gate)

    if c_raw < 22.0:
        s = structural_score * 0.38
    elif c_raw < 38.0:
        # Strong holistic similarity but patchy concept hinge → allow more structural credit.
        damp = 0.64 if sem >= 52.0 else 0.58
        s = structural_score * damp
    elif c_raw < 52.0:
        s = structural_score * 0.72
    else:
        s = structural_score

    # Thin rubric coverage + long answer → curb length inflation (verbose weak answers).
    if (
        coverage_ratio is not None
        and float(coverage_ratio) < 0.22
        and c_raw < 30.0
        and structural_score >= 50.0
    ):
        s *= 0.82

    s = float(np.clip(s, 0.0, STRUCTURAL_CEILING))

    hybrid = (
        WEIGHT_CONCEPT * c
        + WEIGHT_STRUCT * s
        + WEIGHT_SEMANTIC * sem
    )

    if coverage_ratio is not None:
        if coverage_ratio == 0.0:
            hybrid = min(hybrid, COVERAGE_CAP_NONE)
        elif coverage_ratio < 0.40:
            hybrid = min(hybrid, COVERAGE_CAP_40)
        elif coverage_ratio < 0.60:
            hybrid = min(hybrid, COVERAGE_CAP_60)

    cr = float(np.clip(contradiction_ratio, 0.0, 1.0))
    hybrid -= CONTRADICTION_ADDITIVE_MAX * cr

    out = round(float(np.clip(hybrid, 0.0, 100.0)), 2)

    return out
