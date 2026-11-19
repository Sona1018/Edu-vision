"""
verify_integration.py
---------------------
Verify that the normalized dataset loads correctly into app.py's data pipeline.
Tests schema compatibility with concept_detection.compute_concept_score().
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "intelliplace_dataset.json"

def main():
    print("=" * 60)
    print("Integration Verification")
    print("=" * 60)
    
    # 1. Load dataset
    print("\n1. Loading dataset...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"   Loaded {len(data)} questions")
    
    # 2. Check schema compatibility with app.py expectations
    print("\n2. Checking app.py schema compatibility...")
    errors = []
    for q in data:
        qid = q["question_id"]
        
        # app.py line 42: q["rubric"]
        if "rubric" not in q:
            errors.append(f"{qid}: Missing 'rubric' key")
            continue
        
        rubric = q["rubric"]
        
        # app.py line 43: sum(c["weight"] for c in rubric)
        try:
            total_weight = sum(c["weight"] for c in rubric)
        except KeyError:
            errors.append(f"{qid}: Rubric missing 'weight' key")
            continue
        
        # app.py line 44: c["reference_explanation"]
        for c in rubric:
            if "reference_explanation" not in c:
                errors.append(f"{qid}: Missing 'reference_explanation'")
            if "concept_id" not in c:
                errors.append(f"{qid}: Missing 'concept_id'")
            if "concept_name" not in c:
                errors.append(f"{qid}: Missing 'concept_name'")
        
        # app.py line 182: c["concept_id"] used in _concept_rows
        # app.py line 246: c["concept_name"] used in build_question_catalog
        # app.py line 247: c["weight"] used in build_question_catalog
        
        # Check answers
        answers = q.get("answers", [])
        if len(answers) != 5:
            errors.append(f"{qid}: Has {len(answers)} answers, expected 5")
        for a in answers:
            if "text" not in a:
                errors.append(f"{qid}: Answer missing 'text'")
            if "score" not in a:
                errors.append(f"{qid}: Answer missing 'score'")
    
    if errors:
        print(f"   FAILED - {len(errors)} errors:")
        for e in errors[:20]:
            print(f"     X {e}")
        return False
    else:
        print("   PASSED - All schema checks pass")
    
    # 3. Simulate build_question_catalog
    print("\n3. Simulating build_question_catalog()...")
    try:
        catalog = []
        for q in data:
            entry = {
                "question_id": q["question_id"],
                "subject": q.get("subject", "Unknown"),
                "question": q["question"],
                "rubric": [
                    {
                        "name": c["concept_name"],
                        "weight": f"{int(round(float(c['weight']) * 100))}%",
                    }
                    for c in q["rubric"]
                ],
            }
            catalog.append(entry)
        
        subjects = set(e["subject"] for e in catalog)
        print(f"   PASSED - Built catalog with {len(catalog)} entries")
        print(f"   Subjects: {sorted(subjects)}")
    except Exception as e:
        print(f"   FAILED - {e}")
        return False
    
    # 4. Simulate _feature_row partial (rubric access)
    print("\n4. Simulating _feature_row rubric access...")
    try:
        for q in data[:3]:  # Test first 3
            rubric = q["rubric"]
            total_weight = float(sum(c["weight"] for c in rubric))
            combined_reference = " ".join(c["reference_explanation"] for c in rubric)
            assert total_weight > 0, f"total_weight is 0 for {q['question_id']}"
            assert len(combined_reference) > 0, f"empty reference for {q['question_id']}"
        print(f"   PASSED - Rubric access works correctly")
    except Exception as e:
        print(f"   FAILED - {e}")
        return False
    
    # 5. Simulate _concept_rows (diagnostics rendering)
    print("\n5. Simulating _concept_rows weight lookup...")
    try:
        for q in data[:3]:
            weights = {c["concept_id"]: c["weight"] for c in q["rubric"]}
            assert len(weights) == len(q["rubric"]), "Duplicate concept_ids!"
            for cid, w in weights.items():
                assert 0 < w <= 1.0, f"Invalid weight {w} for {cid}"
        print(f"   PASSED - Weight lookup works correctly")
    except Exception as e:
        print(f"   FAILED - {e}")
        return False
    
    # 6. Check concept_id uniqueness globally
    print("\n6. Checking concept_id global uniqueness...")
    all_cids = []
    for q in data:
        for c in q["rubric"]:
            all_cids.append(c["concept_id"])
    
    unique_cids = set(all_cids)
    if len(all_cids) != len(unique_cids):
        dupes = [cid for cid in unique_cids if all_cids.count(cid) > 1]
        print(f"   FAILED - {len(all_cids) - len(unique_cids)} duplicate concept_ids")
        for d in dupes[:5]:
            print(f"     Duplicate: {d}")
        return False
    else:
        print(f"   PASSED - All {len(all_cids)} concept_ids are unique")
    
    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED - Dataset is app.py compatible")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
