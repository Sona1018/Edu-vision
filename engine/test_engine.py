import json

from engine.concept_detection import compute_concept_score
from engine.mastery import (
    WEIGHT_CONCEPT,
    WEIGHT_SEMANTIC,
    WEIGHT_STRUCT,
    compute_contradiction_ratio,
    compute_hybrid_score,
    compute_structural_score,
    soft_coverage_ratio,
)
from engine.scoring import compute_semantic_score

with open("data/questions.json", "r") as f:
    questions = json.load(f)

question = questions[0]

print("\nQuestion:")
print(question["question_text"])

student_answer = input("\nEnter your answer:\n")

rubric = question["rubric"]
total_weight = question.get("total_weight")
if total_weight is None:
    total_weight = sum(float(c["weight"]) for c in rubric)

combined_reference = " ".join(item["reference_explanation"] for item in rubric)

semantic_score, off_topic_gate = compute_semantic_score(
    student_answer,
    combined_reference,
)

concept_score, concept_details = compute_concept_score(
    student_answer,
    rubric,
    total_weight,
)

structural_score = compute_structural_score(student_answer)

gated_concept = concept_score * off_topic_gate

adj_sims = [float(d["adjusted_similarity"]) for d in concept_details]
cov_ratio = float(soft_coverage_ratio(adj_sims)) if adj_sims else 0.0
contra_ratio = compute_contradiction_ratio(concept_details, n_c)

hybrid_score = compute_hybrid_score(
    concept_score=gated_concept,
    structural_score=structural_score,
    semantic_score=semantic_score,
    coverage_ratio=cov_ratio,
    contradiction_ratio=contra_ratio,
    off_topic_gate=float(off_topic_gate),
)

print("\n--- Results ---")
print("Semantic Score:", semantic_score, "| Off-topic gate:", round(off_topic_gate, 3))
print("Concept Score:", round(concept_score, 2), "| Gated:", round(gated_concept, 2))
print("Structural Score:", structural_score)
print("Hybrid Score:", hybrid_score)

print("\n===== Concept Evidence Analysis =====\n")

for detail in concept_details:
    print("Concept:", detail["concept_name"])
    print("Evidence Sentence:")
    print(" ", detail["best_sentence"])
    print("Max sentence SBERT sim:", detail["raw_sbert_sim"])
    print("Combined quality:", detail["combined_quality"])
    print("Entailment:", detail["entailment"], "| Contradiction:", detail["contradiction"])
    print("Depth Score:", detail["depth_score"])
    print("Adjusted Similarity:", detail["adjusted_similarity"])
    print("Weighted Score:", detail["weighted_score"])
    print("----------------------------")

print(
    "\nHybrid weights (concept, struct, semantic):",
    WEIGHT_CONCEPT,
    WEIGHT_STRUCT,
    WEIGHT_SEMANTIC,
)
