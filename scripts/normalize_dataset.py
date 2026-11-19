"""
normalize_dataset.py
--------------------
Production-level normalization of intelliplace_dataset.json.

Performs the following transformations IN-PLACE on the existing dataset:
1. Schema fix: Convert 'concepts' -> 'rubric' with auto-generated concept_id and weight
2. Subject name normalization (ensure full canonical names)
3. Validate: exactly 5 answers per question, scores in [1, 10]
4. Score floor: clamp any score < 1 to 1
5. Detect and report score pattern leakage (no changes, just warnings)
6. Write back to intelliplace_dataset.json

NO NEW DATA IS GENERATED. This script only cleans, normalizes, and restructures.
"""

import json
import sys
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "intelliplace_dataset.json"
BACKUP_PATH = BASE_DIR / "dataset" / "intelliplace_dataset.backup.json"

# Subject name normalization map
SUBJECT_MAP = {
    "CN": "Computer Networks",
    "Computer Networks": "Computer Networks",
    "DBMS": "Database Management Systems",
    "Database Management Systems": "Database Management Systems",
    "Data Structures": "Data Structures",
    "DS": "Data Structures",
    "Operating Systems": "Operating Systems",
    "OS": "Operating Systems",
}


def normalize_subject(raw: str) -> str:
    """Normalize subject name to canonical form."""
    normalized = SUBJECT_MAP.get(raw)
    if normalized is None:
        print(f"  WARNING: Unknown subject '{raw}', keeping as-is")
        return raw
    return normalized


def build_rubric(concepts: list, question_id: str) -> list:
    """
    Convert 'concepts' list to 'rubric' list with:
    - concept_id: {question_id}_C{index}
    - concept_name: from original
    - reference_explanation: from original
    - weight: equal distribution (1.0 / n_concepts)
    """
    n = len(concepts)
    if n == 0:
        return []
    
    weight = round(1.0 / n, 4)
    rubric = []
    for i, concept in enumerate(concepts, start=1):
        rubric.append({
            "concept_id": f"{question_id}_C{i}",
            "concept_name": concept["concept_name"],
            "reference_explanation": concept["reference_explanation"],
            "weight": weight,
        })
    return rubric


def validate_question(q: dict, idx: int) -> list:
    """Validate a single question entry. Returns list of error strings."""
    errors = []
    qid = q.get("question_id", f"UNKNOWN_{idx}")
    
    # Check required fields
    for field in ["question_id", "subject", "question", "rubric", "answers"]:
        if field not in q:
            errors.append(f"{qid}: Missing required field '{field}'")
    
    # Check rubric structure
    if "rubric" in q:
        rubric = q["rubric"]
        if len(rubric) == 0:
            errors.append(f"{qid}: Empty rubric")
        for ci, c in enumerate(rubric):
            for key in ["concept_id", "concept_name", "reference_explanation", "weight"]:
                if key not in c:
                    errors.append(f"{qid}: Rubric concept {ci} missing '{key}'")
        
        # Check weight sum
        total_weight = sum(c.get("weight", 0) for c in rubric)
        if abs(total_weight - 1.0) > 0.05:
            errors.append(f"{qid}: Rubric weights sum to {total_weight:.4f}, expected ~1.0")
    
    # Check answers
    if "answers" in q:
        answers = q["answers"]
        if len(answers) != 5:
            errors.append(f"{qid}: Has {len(answers)} answers, expected 5")
        for ai, a in enumerate(answers):
            if "text" not in a or "score" not in a:
                errors.append(f"{qid}: Answer {ai} missing 'text' or 'score'")
            else:
                score = a["score"]
                if not (1 <= score <= 10):
                    errors.append(f"{qid}: Answer {ai} score={score} outside [1,10]")
    
    return errors


def check_pattern_leakage(data: list) -> dict:
    """Detect score pattern repetition across questions. Returns pattern stats."""
    patterns = Counter()
    for q in data:
        scores = tuple(sorted(a["score"] for a in q.get("answers", [])))
        patterns[scores] += 1
    
    total = len(data)
    leakage_report = {}
    for pattern, count in patterns.most_common(10):
        leakage_report[str(pattern)] = {
            "count": count,
            "pct": round(count / total * 100, 1),
            "flag": "CRITICAL" if count / total > 0.30 else "WARNING" if count / total > 0.15 else "OK"
        }
    return leakage_report


def main():
    print("=" * 60)
    print("IntelliPlace Dataset Normalization")
    print("=" * 60)
    
    # Load
    print(f"\nLoading: {DATASET_PATH}")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} questions")
    
    # Backup
    print(f"\nCreating backup: {BACKUP_PATH}")
    with open(BACKUP_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("  Backup saved")
    
    # === PHASE 1: Schema transformation ===
    print("\n--- Phase 1: Schema Transformation ---")
    schema_fixes = 0
    subject_fixes = 0
    score_fixes = 0
    
    for i, q in enumerate(data):
        qid = q.get("question_id", f"UNK_{i}")
        
        # 1a. Subject normalization
        raw_subject = q.get("subject", "")
        normalized_subject = normalize_subject(raw_subject)
        if normalized_subject != raw_subject:
            print(f"  {qid}: Subject '{raw_subject}' -> '{normalized_subject}'")
            q["subject"] = normalized_subject
            subject_fixes += 1
        
        # 1b. concepts -> rubric conversion
        if "concepts" in q and "rubric" not in q:
            rubric = build_rubric(q["concepts"], qid)
            q["rubric"] = rubric
            del q["concepts"]
            schema_fixes += 1
        elif "rubric" in q and "concepts" in q:
            # Both exist — keep rubric, remove concepts
            del q["concepts"]
            schema_fixes += 1
        elif "rubric" in q:
            # Already has rubric — ensure it has all required fields
            for ci, c in enumerate(q["rubric"]):
                if "concept_id" not in c:
                    c["concept_id"] = f"{qid}_C{ci+1}"
                if "weight" not in c:
                    c["weight"] = round(1.0 / len(q["rubric"]), 4)
        
        # 1c. Score clamping (floor to 1, ceil to 10)
        for a in q.get("answers", []):
            old_score = a["score"]
            a["score"] = max(1, min(10, int(a["score"])))
            if a["score"] != old_score:
                print(f"  {qid}: Score {old_score} -> {a['score']}")
                score_fixes += 1
    
    print(f"\n  Schema fixes: {schema_fixes}")
    print(f"  Subject fixes: {subject_fixes}")
    print(f"  Score fixes: {score_fixes}")
    
    # === PHASE 2: Validation ===
    print("\n--- Phase 2: Validation ---")
    all_errors = []
    for i, q in enumerate(data):
        errors = validate_question(q, i)
        all_errors.extend(errors)
    
    if all_errors:
        print(f"  ERRORS ({len(all_errors)}):")
        for err in all_errors:
            print(f"    ✗ {err}")
    else:
        print("  ✓ All questions pass validation")
    
    # === PHASE 3: Pattern leakage check ===
    print("\n--- Phase 3: Score Pattern Analysis ---")
    
    # Global
    leakage = check_pattern_leakage(data)
    print("  Global patterns (top 10):")
    for pattern, info in leakage.items():
        flag = info["flag"]
        icon = "✗" if flag == "CRITICAL" else "⚠" if flag == "WARNING" else "✓"
        print(f"    {icon} {pattern}: {info['count']} times ({info['pct']}%) [{flag}]")
    
    # Per-subject
    subjects = set(q["subject"] for q in data)
    for subj in sorted(subjects):
        subj_data = [q for q in data if q["subject"] == subj]
        subj_leakage = check_pattern_leakage(subj_data)
        worst = next(iter(subj_leakage.values()), {"pct": 0, "flag": "OK"})
        print(f"\n  {subj} ({len(subj_data)} questions):")
        for pattern, info in list(subj_leakage.items())[:3]:
            flag = info["flag"]
            icon = "✗" if flag == "CRITICAL" else "⚠" if flag == "WARNING" else "✓"
            print(f"    {icon} {pattern}: {info['count']} times ({info['pct']}%) [{flag}]")
    
    # === PHASE 4: Summary ===
    print("\n--- Phase 4: Summary ---")
    subjects_summary = Counter(q["subject"] for q in data)
    for subj, count in sorted(subjects_summary.items()):
        print(f"  {subj}: {count} questions")
    print(f"  Total: {len(data)} questions")
    
    # Schema check
    all_have_rubric = all("rubric" in q for q in data)
    none_have_concepts = all("concepts" not in q for q in data)
    all_have_5_answers = all(len(q.get("answers", [])) == 5 for q in data)
    all_scores_valid = all(1 <= a["score"] <= 10 for q in data for a in q.get("answers", []))
    all_weights_valid = all(
        abs(sum(c["weight"] for c in q["rubric"]) - 1.0) < 0.05
        for q in data if "rubric" in q
    )
    
    print(f"\n  ✓ All have 'rubric' key: {all_have_rubric}")
    print(f"  ✓ None have 'concepts' key: {none_have_concepts}")
    print(f"  ✓ All have 5 answers: {all_have_5_answers}")
    print(f"  ✓ All scores in [1,10]: {all_scores_valid}")
    print(f"  ✓ All rubric weights ~1.0: {all_weights_valid}")
    
    if all_errors:
        print(f"\n  ✗ {len(all_errors)} validation errors remain — NOT writing output")
        sys.exit(1)
    
    # === WRITE OUTPUT ===
    print(f"\nWriting normalized dataset to: {DATASET_PATH}")
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("  ✓ Done!")
    
    # Final byte count
    size = DATASET_PATH.stat().st_size
    print(f"  Output size: {size:,} bytes ({size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
