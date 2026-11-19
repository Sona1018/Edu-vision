"""Quick audit of intelliplace_dataset.json"""
import json
from pathlib import Path
from collections import Counter

DATASET = Path(__file__).resolve().parent.parent / "dataset" / "intelliplace_dataset.json"
data = json.load(open(DATASET, "r", encoding="utf-8"))

print(f"Total questions: {len(data)}")
print()

# All question IDs + subjects
for q in data:
    qid = q["question_id"]
    subj = q.get("subject", "NONE")
    n_ans = len(q.get("answers", []))
    scores = sorted([a["score"] for a in q.get("answers", [])])
    has_rubric = "rubric" in q
    has_concepts = "concepts" in q
    schema = "rubric" if has_rubric else ("concepts" if has_concepts else "NONE")
    print(f"  {qid:12s} | {subj:30s} | {n_ans} ans | scores={scores} | schema={schema}")

print()
# Summary
subjects = Counter(q.get("subject", "NONE") for q in data)
print("Subject counts:", dict(subjects))

# Check for any OS-like content in questions
for q in data:
    qt = q.get("question", "").lower()
    if any(kw in qt for kw in ["operating system", "process", "deadlock", "virtual memory", "semaphore", "paging", "cpu scheduling", "thread"]):
        print(f"  Potential OS content: {q['question_id']} -> {q['subject']} -> {q['question'][:80]}")
