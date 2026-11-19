"""
concept_analysis.py
-------------------
Categorises concept mastery scores into weak/moderate/strong buckets.

DO YOU NEED THIS FILE?
  YES — it is used in evaluate_engine.py to produce the
  "WEAK / MODERATE / STRONG CONCEPTS" section of the output.

  This is the feedback layer for the adaptive learning feature.
  A teacher or adaptive system can use these categories to:
  - Show students which OS concepts they need to review
  - Prioritise questions on weak concepts in future sessions
  - Track improvement over time as mastery updates

  Nothing in this file affects scoring. It is pure post-processing
  of the mastery vector for human-readable output.

THRESHOLDS (from MasteryVector):
  Weak     < 0.40  → Student frequently gets this concept wrong
  Moderate 0.40–0.70 → Student has partial understanding
  Strong   > 0.70  → Student reliably explains this concept correctly
"""


def analyze_mastery(mastery_vector: dict):
    """
    Split a mastery vector dict into three quality buckets.

    Parameters
    ----------
    mastery_vector : dict
        {concept_id: mastery_score} where mastery_score ∈ [0, 1]

    Returns
    -------
    weak     : list of (concept_id, score) tuples — mastery < 0.40
    moderate : list of (concept_id, score) tuples — 0.40 ≤ mastery < 0.70
    strong   : list of (concept_id, score) tuples — mastery ≥ 0.70
    """
    weak     = []
    moderate = []
    strong   = []

    for concept, score in mastery_vector.items():
        if score < 0.40:
            weak.append((concept, score))
        elif score < 0.70:
            moderate.append((concept, score))
        else:
            strong.append((concept, score))

    # Sort each bucket by score for readability
    weak.sort(key=lambda x: x[1])
    moderate.sort(key=lambda x: x[1])
    strong.sort(key=lambda x: x[1], reverse=True)

    return weak, moderate, strong