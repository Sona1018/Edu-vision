import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import json

from engine.concept_detection import compute_concept_score
from engine.scoring import compute_semantic_score
from engine.mastery import (
    compute_hybrid_score,
    compute_structural_score,
    soft_coverage_ratio,
    compute_contradiction_ratio
)

with open('dataset/marketing_reference_dataset.json', encoding='utf-8') as f:
    dataset = json.load(f)

print('======================================================================')
print('TEST RUN: EVALUATING MARKETING ANSWERS WITH INTELLIPLACE ENGINE')
print('======================================================================\n')

total_diff = 0.0
total_tested = 0

for q in dataset[:2]:  # Q1 and Q2
    qid = q['question_id']
    q_text = q['question']
    rubric = q['rubric']
    total_w = sum(c['weight'] for c in rubric)
    comb_ref = ' '.join(c['reference_explanation'] for c in rubric)
    ans_list = q['answers']
    
    print(f"📘 {qid}: {q_text[:65]}...")
    print(f"   Rubric Concepts: {len(rubric)} | Total Available Answers: {len(ans_list)}")
    print('   ------------------------------------------------------------------')
    
    for i, ans in enumerate(ans_list[:4], 1):  # Test 4 students per question
        text = ans['text']
        human_score = ans['score']
        
        sem_score, off_topic_gate = compute_semantic_score(text, comb_ref)
        c_score, details = compute_concept_score(text, rubric, total_w)
        gated_c = c_score * off_topic_gate
        struct_score = compute_structural_score(text)
        
        adjs = [float(d['adjusted_similarity']) for d in details]
        cov_ratio = float(soft_coverage_ratio(adjs)) if adjs else 0.0
        contra_ratio = compute_contradiction_ratio(details, len(rubric))
        
        hybrid = compute_hybrid_score(gated_c, struct_score, sem_score, cov_ratio, contra_ratio, off_topic_gate)
        ai_score_10 = round(hybrid / 10.0, 2)
        diff = round(abs(ai_score_10 - human_score), 2)
        
        total_diff += diff
        total_tested += 1
        
        words = len(text.split())
        print(f"   Student {i} ({words} words) -> AI Score: {ai_score_10}/10 | Target: {human_score}/10 | Error: ±{diff}")
    print()

mae = round(total_diff / total_tested, 2) if total_tested > 0 else 0
print('======================================================================')
print(f"🎯 Average Test Error (MAE): ±{mae} marks out of 10")
print('======================================================================')
