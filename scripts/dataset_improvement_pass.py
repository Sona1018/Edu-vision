import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


def _norm(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _sentence_chunks(text: str) -> list[str]:
    chunks = re.split(r"[.!?]\s+", text.strip())
    return [c.strip() for c in chunks if len(c.strip()) >= 30]


def audit_dataset(dataset: list[dict]) -> dict:
    per_question_exact_dupes = {}
    all_answers: list[tuple[str, str, float]] = []
    sentence_counter: Counter[str] = Counter()
    score_band_flags = []

    for q in dataset:
        qid = q["question_id"]
        seen = {}
        dupes = 0
        for ans in q["answers"]:
            text = ans["text"]
            score = float(ans["score"])
            norm = _norm(text)
            all_answers.append((qid, norm, score))
            if norm in seen:
                dupes += 1
            else:
                seen[norm] = True

            for s in _sentence_chunks(text):
                sentence_counter[_norm(s)] += 1

            words = len(text.split())
            if 4 <= score <= 7:
                if words < 22 or words > 240:
                    score_band_flags.append(
                        {
                            "question_id": qid,
                            "score": score,
                            "word_count": words,
                            "preview": text[:120],
                        }
                    )
        per_question_exact_dupes[qid] = dupes

    near_dupes = []
    for i in range(len(all_answers)):
        qid_i, t_i, s_i = all_answers[i]
        for j in range(i + 1, len(all_answers)):
            qid_j, t_j, s_j = all_answers[j]
            if qid_i == qid_j:
                continue
            if abs(len(t_i) - len(t_j)) > 80:
                continue
            sim = SequenceMatcher(None, t_i, t_j).ratio()
            if sim >= 0.93:
                near_dupes.append(
                    {
                        "q1": qid_i,
                        "q2": qid_j,
                        "score1": s_i,
                        "score2": s_j,
                        "similarity": round(float(sim), 4),
                        "preview": t_i[:120],
                    }
                )
                if len(near_dupes) >= 120:
                    break
        if len(near_dupes) >= 120:
            break

    repeated_templates = [
        {"text": t[:180], "count": c}
        for t, c in sentence_counter.most_common(80)
        if c >= 4
    ]

    return {
        "questions": len(dataset),
        "rows_before": sum(len(q["answers"]) for q in dataset),
        "per_question_exact_duplicates": per_question_exact_dupes,
        "near_duplicates_cross_question": near_dupes,
        "repeated_template_phrases": repeated_templates,
        "label_consistency_flags_4_to_7": score_band_flags[:200],
    }


def _adversarial_answer(q: dict) -> dict:
    names = [c["concept_name"] for c in q["rubric"]][:3]
    joined = ", ".join(names)
    text = (
        f"Some students assume this topic is mostly about memorizing terms. "
        f"In that view, {joined} are optional and usually not required for correct system behavior. "
        f"So practical performance comes mainly from hardware speed and not from these operating-system mechanisms."
    )
    return {"text": text, "score": 2}


def _cross_topic_distractor(q: dict) -> dict:
    text = (
        "This answer focuses on user-interface convenience such as icons, themes, and desktop shortcuts. "
        "Most system quality comes from visual design and app layout choices. "
        "As long as menus are clear and navigation is easy, low-level operating-system details are not very important."
    )
    return {"text": text, "score": 1}


def _concise_high(q: dict) -> dict:
    names = [c["concept_name"] for c in q["rubric"]][:3]
    text = (
        f"A correct explanation should directly cover {names[0]}, {names[1]}, and {names[2]}. "
        "Together these show why the OS coordinates execution, resources, and safe interaction with hardware."
    )
    return {"text": text, "score": 8}


def apply_cleanup_and_augmentation(dataset: list[dict]) -> tuple[list[dict], dict]:
    added = 0
    removed_duplicates = 0
    out = []
    for q in dataset:
        answers = q["answers"]
        seen = set()
        cleaned = []
        for a in answers:
            norm = _norm(a["text"])
            if norm in seen:
                removed_duplicates += 1
                continue
            seen.add(norm)
            cleaned.append(a)

        for new_ans in (_adversarial_answer(q), _cross_topic_distractor(q), _concise_high(q)):
            norm = _norm(new_ans["text"])
            if norm not in seen:
                cleaned.append(new_ans)
                seen.add(norm)
                added += 1

        q_new = dict(q)
        q_new["answers"] = cleaned
        out.append(q_new)

    return out, {"rows_after": sum(len(q["answers"]) for q in out), "added_rows": added, "removed_duplicates": removed_duplicates}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and improve IntelliPlace dataset quality.")
    parser.add_argument("--dataset", default="dataset/intelliplace_dataset.json")
    parser.add_argument("--report-out", default="artifacts/dataset_audit_report.json")
    parser.add_argument("--output-dataset", default="dataset/intelliplace_dataset.json")
    parser.add_argument("--apply", action="store_true", help="Apply cleanup + augmentation and write dataset.")
    args, _unknown = parser.parse_known_args()

    src = Path(args.dataset)
    dataset = json.loads(src.read_text(encoding="utf-8"))
    audit_before = audit_dataset(dataset)
    report = {"before": audit_before}

    if args.apply:
        updated, stats = apply_cleanup_and_augmentation(dataset)
        report["changes"] = stats
        report["after"] = audit_dataset(updated)
        Path(args.output_dataset).write_text(json.dumps(updated, indent=2), encoding="utf-8")
        print(
            f"Applied dataset cleanup+augmentation. Added {stats['added_rows']} rows, "
            f"removed {stats['removed_duplicates']} exact duplicates."
        )
        print(f"Wrote updated dataset to: {args.output_dataset}")
    else:
        print("Audit-only mode (no dataset changes).")

    out_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote audit report to: {out_path}")


if __name__ == "__main__":
    main()
