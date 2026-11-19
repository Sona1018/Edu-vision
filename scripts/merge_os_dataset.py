"""
merge_os_dataset.py
-------------------
Safe integration of os.json into the existing normalized intelliplace_dataset.json.

Steps:
1. Load existing dataset (CN, DBMS, DS) — already normalized with rubric schema
2. Load os.json — uses 'concepts' schema, needs normalization
3. Validate os.json: 5 answers/question, scores in [1,10], no duplicates
4. Normalize os.json: concepts -> rubric, add concept_id/weight, clamp scores
5. Check for question_id collisions
6. Check score pattern leakage in OS data
7. Merge and write final dataset
8. Run full validation on merged output
"""

import json
import sys
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "intelliplace_dataset.json"
OS_PATH = BASE_DIR / "dataset" / "os.json"
BACKUP_PATH = BASE_DIR / "dataset" / "intelliplace_dataset.pre_os_merge.backup.json"

# ─── Helpers ────────────────────────────────────────────────────────────────

def build_rubric(concepts, question_id):
    n = len(concepts)
    if n == 0:
        return []
    weight = round(1.0 / n, 4)
    rubric = []
    for i, c in enumerate(concepts, start=1):
        rubric.append({
            "concept_id": f"{question_id}_C{i}",
            "concept_name": c["concept_name"],
            "reference_explanation": c["reference_explanation"],
            "weight": weight,
        })
    return rubric


def validate_question(q, idx):
    errors = []
    qid = q.get("question_id", f"UNKNOWN_{idx}")
    for field in ["question_id", "subject", "question", "rubric", "answers"]:
        if field not in q:
            errors.append(f"{qid}: Missing field '{field}'")
    if "rubric" in q:
        rubric = q["rubric"]
        if len(rubric) == 0:
            errors.append(f"{qid}: Empty rubric")
        for ci, c in enumerate(rubric):
            for key in ["concept_id", "concept_name", "reference_explanation", "weight"]:
                if key not in c:
                    errors.append(f"{qid}: Rubric[{ci}] missing '{key}'")
        total_w = sum(c.get("weight", 0) for c in rubric)
        if abs(total_w - 1.0) > 0.05:
            errors.append(f"{qid}: Weights sum to {total_w:.4f}")
    if "answers" in q:
        answers = q["answers"]
        if len(answers) != 5:
            errors.append(f"{qid}: Has {len(answers)} answers (expected 5)")
        for ai, a in enumerate(answers):
            if "text" not in a or "score" not in a:
                errors.append(f"{qid}: Answer[{ai}] missing 'text' or 'score'")
            elif not (1 <= a["score"] <= 10):
                errors.append(f"{qid}: Answer[{ai}] score={a['score']} outside [1,10]")
    return errors


def check_patterns(data, label):
    patterns = Counter()
    for q in data:
        key = tuple(sorted(a["score"] for a in q.get("answers", [])))
        patterns[key] += 1
    total = len(data)
    print(f"\n  {label} score patterns (top 5):")
    for pattern, count in patterns.most_common(5):
        pct = round(count / total * 100, 1)
        flag = "CRITICAL" if pct > 30 else "WARNING" if pct > 15 else "OK"
        icon = "X" if flag == "CRITICAL" else "!" if flag == "WARNING" else "."
        print(f"    [{icon}] {pattern}: {count}x ({pct}%) [{flag}]")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("OS Dataset Integration")
    print("=" * 60)

    # 1. Load existing dataset
    print("\n1. Loading existing dataset...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        existing = json.load(f)
    existing_ids = {q["question_id"] for q in existing}
    existing_subjects = Counter(q["subject"] for q in existing)
    print(f"   {len(existing)} questions loaded")
    for subj, cnt in sorted(existing_subjects.items()):
        print(f"     {subj}: {cnt}")

    # 2. Load OS dataset
    print("\n2. Loading os.json...")
    with open(OS_PATH, "r", encoding="utf-8") as f:
        os_data = json.load(f)
    print(f"   {len(os_data)} OS questions loaded")

    # 3. Pre-normalization audit of os.json
    print("\n3. Pre-normalization audit of os.json...")
    os_issues = {"score_clamps": 0, "schema_fixes": 0, "subject_fixes": 0}

    for i, q in enumerate(os_data):
        qid = q.get("question_id", f"OS_UNK_{i}")

        # Schema: concepts -> rubric
        if "concepts" in q and "rubric" not in q:
            q["rubric"] = build_rubric(q["concepts"], qid)
            del q["concepts"]
            os_issues["schema_fixes"] += 1
        elif "rubric" in q and "concepts" in q:
            del q["concepts"]
            os_issues["schema_fixes"] += 1
        elif "rubric" in q:
            for ci, c in enumerate(q["rubric"]):
                if "concept_id" not in c:
                    c["concept_id"] = f"{qid}_C{ci+1}"
                if "weight" not in c:
                    c["weight"] = round(1.0 / len(q["rubric"]), 4)

        # Subject normalization
        raw_subj = q.get("subject", "")
        if raw_subj != "Operating Systems":
            q["subject"] = "Operating Systems"
            os_issues["subject_fixes"] += 1
            print(f"   Subject fix: {qid}: '{raw_subj}' -> 'Operating Systems'")

        # Score clamping [1, 10]
        for a in q.get("answers", []):
            old = a["score"]
            a["score"] = max(1, min(10, int(a["score"])))
            if a["score"] != old:
                print(f"   Score clamp: {qid}: {old} -> {a['score']}")
                os_issues["score_clamps"] += 1

    print(f"\n   Schema fixes: {os_issues['schema_fixes']}")
    print(f"   Subject fixes: {os_issues['subject_fixes']}")
    print(f"   Score clamps: {os_issues['score_clamps']}")

    # 4. Validate OS data post-normalization
    print("\n4. Validating OS data post-normalization...")
    all_errors = []
    for i, q in enumerate(os_data):
        errs = validate_question(q, i)
        all_errors.extend(errs)

    if all_errors:
        print(f"   ERRORS ({len(all_errors)}):")
        for e in all_errors:
            print(f"     X {e}")
        sys.exit(1)
    else:
        print("   PASSED - All OS questions valid")

    # 5. Check for ID collisions
    print("\n5. Checking for question_id collisions...")
    os_ids = {q["question_id"] for q in os_data}
    collisions = existing_ids & os_ids
    if collisions:
        print(f"   COLLISION detected: {collisions}")
        sys.exit(1)
    else:
        print(f"   PASSED - No collisions ({len(os_ids)} OS IDs, {len(existing_ids)} existing IDs)")

    # 6. Score pattern analysis
    print("\n6. Score pattern analysis...")
    check_patterns(os_data, "OS")
    check_patterns(existing, "Existing (CN+DBMS+DS)")

    # 7. Merge
    print("\n7. Merging datasets...")
    # Backup existing
    print(f"   Backup: {BACKUP_PATH}")
    with open(BACKUP_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    merged = existing + os_data
    print(f"   Merged: {len(existing)} + {len(os_data)} = {len(merged)} questions")

    # 8. Full validation on merged dataset
    print("\n8. Full validation on merged dataset...")
    merged_errors = []
    for i, q in enumerate(merged):
        errs = validate_question(q, i)
        merged_errors.extend(errs)

    if merged_errors:
        print(f"   ERRORS ({len(merged_errors)}):")
        for e in merged_errors[:20]:
            print(f"     X {e}")
        sys.exit(1)
    else:
        print("   PASSED - All merged questions valid")

    # Check global concept_id uniqueness
    all_cids = [c["concept_id"] for q in merged for c in q["rubric"]]
    if len(all_cids) != len(set(all_cids)):
        dupes = [cid for cid in set(all_cids) if all_cids.count(cid) > 1]
        print(f"   DUPLICATE concept_ids: {dupes[:5]}")
        sys.exit(1)
    else:
        print(f"   PASSED - All {len(all_cids)} concept_ids unique")

    # Final subject summary
    final_subjects = Counter(q["subject"] for q in merged)
    print("\n   Final dataset composition:")
    for subj, cnt in sorted(final_subjects.items()):
        print(f"     {subj}: {cnt}")
    print(f"     TOTAL: {len(merged)}")

    # Invariants
    all_rubric = all("rubric" in q for q in merged)
    no_concepts = all("concepts" not in q for q in merged)
    all_5 = all(len(q["answers"]) == 5 for q in merged)
    all_valid_scores = all(1 <= a["score"] <= 10 for q in merged for a in q["answers"])

    print(f"\n   All have 'rubric': {all_rubric}")
    print(f"   None have 'concepts': {no_concepts}")
    print(f"   All have 5 answers: {all_5}")
    print(f"   All scores in [1,10]: {all_valid_scores}")

    if not (all_rubric and no_concepts and all_5 and all_valid_scores):
        print("\n   INVARIANT FAILURE - aborting write")
        sys.exit(1)

    # 9. Write merged dataset
    print(f"\n9. Writing merged dataset to: {DATASET_PATH}")
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    size = DATASET_PATH.stat().st_size
    print(f"   Done! Size: {size:,} bytes ({size/1024:.1f} KB)")
    print("\n" + "=" * 60)
    print("OS INTEGRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
