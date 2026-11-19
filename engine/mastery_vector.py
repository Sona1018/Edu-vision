"""
mastery_vector.py
-----------------
Tracks per-concept mastery using exponential moving average.

DO YOU NEED THIS FILE?
  YES — for the secondary adaptive learning feature.
  It is NOT part of the scoring pipeline.
  It runs AFTER scoring to track which concepts a student is weak on
  over multiple attempts, enabling personalised feedback.

  If you only want the evaluation score, you can skip calling
  mastery_system.update() and mastery_system.get_mastery().
  But keeping it adds value with zero cost to scoring quality.

HOW IT WORKS:
  Each concept has a running mastery score initialised to 0.
  After each answer is evaluated, the adjusted_similarity for each
  concept is used to update mastery via exponential moving average:

    new_mastery = alpha * new_score + (1 - alpha) * old_mastery

  alpha=0.4 means recent performance has 40% weight, history has 60%.
  This smooths out single lucky/unlucky answers.

  concept_analysis.py categorises concepts into:
    Weak     (mastery < 0.40): student needs review
    Moderate (0.40 - 0.70):   student has basic understanding
    Strong   (> 0.70):        student has solid grasp

  NOTE: mastery uses remapped adjusted_similarity values (post-sigmoid),
  not raw SBERT values. This means mastery correctly reflects whether
  the student's explanation was actually correct, not just semantically
  close in vocabulary.
"""


class MasteryVector:

    def __init__(self, alpha: float = 0.4):
        """
        Parameters
        ----------
        alpha : float
            Learning rate for exponential moving average.
            0.4 = recent performance weighted at 40%.
        """
        self.alpha  = alpha
        self.mastery: dict = {}

    def update(self, concept_scores: dict):
        """
        Update mastery for each concept in the provided dict.

        Parameters
        ----------
        concept_scores : dict
            {concept_id: adjusted_similarity_score}
            adjusted_similarity is the post-sigmoid, post-NLI score in [0, 1].

        Example
        -------
        mastery.update({
            "C1": 0.72,
            "C2": 0.31,
            "C3": 0.85,
        })
        """
        for concept_id, score in concept_scores.items():
            old_mastery = self.mastery.get(concept_id, 0.0)
            new_mastery = self.alpha * score + (1 - self.alpha) * old_mastery
            self.mastery[concept_id] = round(new_mastery, 3)

    def get_mastery(self) -> dict:
        """Return the current mastery vector."""
        return self.mastery