"""
test_engine_os.py
-----------------
Tests if the concept_detection engine can correctly process an OS question 
from the newly integrated dataset.
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from engine.concept_detection import compute_concept_score

DATASET_PATH = BASE_DIR / "dataset" / "intelliplace_dataset.json"

def main():
    print("Loading dataset...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    os_questions = [q for q in dataset if q["subject"] == "Operating Systems"]
    if not os_questions:
        print("No OS questions found!")
        sys.exit(1)
        
    test_q = os_questions[0]
    sample_answer = test_q["answers"][0]["text"]
    
    print(f"Testing engine with question: {test_q['question_id']}")
    print(f"Question: {test_q['question']}")
    print(f"Answer: {sample_answer[:100]}...")
    
    try:
        # Run engine concept detection
        result = compute_concept_score(
            question_text=test_q["question"],
            answer_text=sample_answer,
            rubric=test_q["rubric"]
        )
        print("\nEngine Output:")
        print(f"Final Score: {result['final_score']}")
        print(f"Base Score: {result['base_score']}")
        print(f"Concepts evaluated: {len(result['concept_results'])}")
        
        for c in result['concept_results']:
            print(f"  - {c['concept_name']}: {c['status']} (Sim: {c.get('semantic_similarity', 0):.2f})")
            
        print("\nSUCCESS: Engine scoring runs correctly for OS questions!")
        sys.exit(0)
    except Exception as e:
        import traceback
        print(f"\nERROR running engine: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
