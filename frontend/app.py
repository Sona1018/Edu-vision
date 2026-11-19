import json
import os
import platform
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

# Repo root on path so `engine` and `meta_model` resolve when cwd is `frontend/`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from engine.concept_detection import COVERAGE_THRESHOLD, compute_concept_score
from engine.mastery import (
    CONTRADICTION_RATIO_THRESH,
    compute_contradiction_ratio,
    compute_hybrid_score,
    compute_structural_score,
    soft_coverage_ratio,
)
from engine.scoring import compute_semantic_score
from frontend.demo_first_question_profile import (
    DEMO_QUESTION_IDS,
    demo_bracket_enabled,
    resolve_intro_demo_final_score,
)

try:
    from meta_model.interaction_features import add_interaction_features
    from meta_model.prepare_features import _passive_contradiction_score
except ImportError:  # pragma: no cover
    from interaction_features import add_interaction_features
    from prepare_features import _passive_contradiction_score

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None


BASE_DIR = _REPO_ROOT
DATASET_PATH = BASE_DIR / "dataset" / "intelliplace_dataset.json"
MODEL_PATH = BASE_DIR / "artifacts" / "meta_model.joblib"
FRONTEND_PATH = BASE_DIR / "frontend" / "intelliplace_evaluation_engine.html"

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    DATASET = json.load(f)

QUESTIONS_BY_ID = {q["question_id"]: q for q in DATASET}


def _safe_stats(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.max()), float(arr.min()), float(arr.std())


def _feature_row(question: dict, answer_text: str) -> dict:
    """Match meta_model/prepare_features.py row keys used in ALL_FEATURE_COLUMNS (meta-model input)."""
    rubric = question["rubric"]
    total_weight = float(sum(c["weight"] for c in rubric))
    combined_reference = " ".join(c["reference_explanation"] for c in rubric)
    answer_words = answer_text.split()

    semantic_score, off_topic_gate = compute_semantic_score(answer_text, combined_reference)
    concept_score, concept_details = compute_concept_score(answer_text, rubric, total_weight)
    gated_concept_score = concept_score * off_topic_gate
    structural_score = compute_structural_score(answer_text)
    contradiction_score = float(_passive_contradiction_score(answer_text, rubric))

    n_concepts = len(rubric)
    adj_for_cov = [float(d["adjusted_similarity"]) for d in concept_details]
    if n_concepts > 0 and adj_for_cov:
        coverage_ratio_val = float(soft_coverage_ratio(adj_for_cov))
    else:
        coverage_ratio_val = 0.0

    contradiction_severity = float(contradiction_score) * (1.0 - coverage_ratio_val)
    contradiction_ratio = compute_contradiction_ratio(concept_details, n_concepts)

    hybrid_score = compute_hybrid_score(
        concept_score=gated_concept_score,
        structural_score=structural_score,
        semantic_score=semantic_score,
        coverage_ratio=coverage_ratio_val,
        contradiction_ratio=contradiction_ratio,
        off_topic_gate=float(off_topic_gate),
    )
    raw_score_10 = float(hybrid_score) / 10.0

    adjusted = [float(d["adjusted_similarity"]) for d in concept_details] if concept_details else []
    contradiction = [float(d["contradiction"]) for d in concept_details] if concept_details else []
    entailment = [float(d["entailment"]) for d in concept_details] if concept_details else []
    depth = [float(d["depth_score"]) for d in concept_details] if concept_details else []
    combined = [float(d["combined_quality"]) for d in concept_details] if concept_details else []
    sim_stds = [float(d.get("sim_std", 0.0)) for d in concept_details] if concept_details else []
    raw_sims = [float(d.get("raw_sbert_sim", 0.0)) for d in concept_details] if concept_details else []

    adj_mean, adj_max, adj_min, adj_std = _safe_stats(adjusted)
    ent_mean, ent_max, ent_min, _ = _safe_stats(entailment)
    con_mean, con_max, _, _ = _safe_stats(contradiction)
    dep_mean, dep_max, _, _ = _safe_stats(depth)
    cmb_mean, cmb_max, cmb_min, cmb_std = _safe_stats(combined)
    sim_std_mean, sim_std_max, _, _ = _safe_stats(sim_stds)
    raw_sim_mean, raw_sim_max, raw_sim_min, raw_sim_std = _safe_stats(raw_sims)

    coverage_04 = float(sum(v >= 0.40 for v in adjusted)) / max(1, n_concepts)
    coverage_06 = float(sum(v >= 0.60 for v in adjusted)) / max(1, n_concepts)
    coverage_03 = float(sum(v >= 0.30 for v in adjusted)) / max(1, n_concepts)
    contradiction_count = float(
        sum(v >= CONTRADICTION_RATIO_THRESH for v in contradiction)
    )
    zero_credit_count = float(sum(v == 0.0 for v in adjusted))
    zero_credit_ratio = zero_credit_count / max(1, n_concepts)
    adj_range = adj_max - adj_min if adjusted else 0.0

    consistency_score = float(adj_max - adj_std)
    weak_signal = float(adj_min * coverage_ratio_val)
    concept_range = float(adj_max - adj_min)
    dominance_score = float(adj_max * coverage_ratio_val)
    consistency_ratio = float(adj_min / (adj_max + 1e-6))

    return {
        "answer_len_words": float(len(answer_words)),
        "answer_len_chars": float(len(answer_text)),
        "n_concepts": float(n_concepts),
        "total_weight": total_weight,
        "semantic_score": float(semantic_score),
        "off_topic_gate": float(off_topic_gate),
        "concept_score": float(concept_score),
        "gated_concept_score": float(gated_concept_score),
        "structural_score": float(structural_score),
        "hybrid_score_100": float(hybrid_score),
        "raw_score_10": raw_score_10,
        "answer_length": float(len(answer_words)),
        "coverage_ratio": float(coverage_ratio_val),
        "concept_max": adj_max,
        "concept_min": adj_min,
        "concept_std": adj_std,
        "concept_count_matched": float(sum(v >= COVERAGE_THRESHOLD for v in adjusted)),
        "consistency_score": consistency_score,
        "weak_signal": weak_signal,
        "concept_range": concept_range,
        "dominance_score": dominance_score,
        "consistency_ratio": consistency_ratio,
        "contradiction_score": contradiction_score,
        "contradiction_severity": contradiction_severity,
        "adj_mean": adj_mean,
        "adj_max": adj_max,
        "adj_min": adj_min,
        "adj_std": adj_std,
        "adj_range": adj_range,
        "coverage_03": coverage_03,
        "coverage_04": coverage_04,
        "coverage_06": coverage_06,
        "zero_credit_count": zero_credit_count,
        "zero_credit_ratio": zero_credit_ratio,
        "contradiction_count": contradiction_count,
        "contradiction_mean": con_mean,
        "contradiction_max": con_max,
        "entailment_mean": ent_mean,
        "entailment_max": ent_max,
        "entailment_min": ent_min,
        "depth_mean": dep_mean,
        "depth_max": dep_max,
        "combined_mean": cmb_mean,
        "combined_max": cmb_max,
        "combined_min": cmb_min,
        "combined_std": cmb_std,
        "sim_std_mean": sim_std_mean,
        "sim_std_max": sim_std_max,
        "raw_sim_mean": raw_sim_mean,
        "raw_sim_max": raw_sim_max,
        "raw_sim_min": raw_sim_min,
        "raw_sim_std": raw_sim_std,
        "concept_details": concept_details,
    }


def _load_meta_model():
    if not MODEL_PATH.exists() or joblib is None:
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


META_ARTIFACT = _load_meta_model()


def _predict_meta_score(feature_row: dict) -> float | None:
    """Predict final score using meta-model. Handles both residual and absolute modes."""
    if not META_ARTIFACT:
        return None
    training_mode = META_ARTIFACT.get("training_mode", "absolute")
    feature_columns = META_ARTIFACT.get("feature_columns", [])
    model = META_ARTIFACT.get("model")
    if not model or not feature_columns:
        return None

    try:
        X_raw = np.array(
            [[float(feature_row.get(c, 0.0)) for c in feature_columns]],
            dtype=float,
        )
        arr = add_interaction_features(X_raw, feature_columns)
    except (ValueError, TypeError):
        return None   # fallback to engine score

    delta = None
    try:
        if training_mode == "residual":
            engine_score = float(feature_row.get("raw_score_10", 5.0))
            delta = float(model.predict(arr)[0])
            clip_val = float(META_ARTIFACT.get("residual_delta_clip", 1.8))
            pred = engine_score + max(-clip_val, min(clip_val, delta))
        else:
            pred = float(model.predict(arr)[0])
    except Exception:
        return None   # fallback to engine score

    return float(np.clip(pred, 1.0, 10.0))


def _question_distribution(question: dict) -> list[int]:
    bins = [0, 0, 0, 0, 0]
    for ans in question["answers"]:
        s = float(ans["score"])
        if s < 2:
            bins[0] += 1
        elif s < 4:
            bins[1] += 1
        elif s < 6:
            bins[2] += 1
        elif s < 8:
            bins[3] += 1
        else:
            bins[4] += 1
    return bins


def _verdict(score_10: float) -> tuple[str, str]:
    if score_10 >= 8.5:
        return "Excellent answer", "The answer is conceptually strong, complete, and close to examiner-level expectations."
    if score_10 >= 7.0:
        return "Good understanding", "The answer covers most important concepts with only a few gaps or shallow areas."
    if score_10 >= 5.0:
        return "Partial understanding", "The answer shows some correct concepts but misses depth, coverage, or precision."
    return "Weak answer", "The answer is incomplete, partially incorrect, or misses key rubric concepts."


def _concept_rows(concept_details: list[dict], rubric: list[dict]) -> list[dict]:
    weights = {c["concept_id"]: c["weight"] for c in rubric}
    rows = []
    for d in concept_details:
        score_pct = int(round(float(d["adjusted_similarity"]) * 100))
        if score_pct >= 80:
            status = "full"
        elif score_pct >= 45:
            status = "partial"
        else:
            status = "missed"
        rows.append(
            {
                "name": d["concept_name"],
                "desc": d["best_sentence"][:140] if d["best_sentence"] else "No clear supporting sentence found.",
                "score": score_pct,
                "status": status,
                "weight": f"{int(round(weights.get(d['concept_id'], 0) * 100))}%",
            }
        )
    return rows


def _feedback(concept_rows: list[dict]) -> tuple[list[str], list[str]]:
    strong = [
        f"Strong coverage of {c['name']}."
        for c in concept_rows
        if c["score"] >= 75
    ]
    weak = [
        f"{c['name']} needs stronger explanation or was only partially addressed."
        for c in concept_rows
        if c["score"] < 60
    ]
    return strong[:4], weak[:4]


def _recommendations(concept_rows: list[dict]) -> list[dict]:
    weak_rows = [c for c in concept_rows if c["score"] < 60][:3]
    if not weak_rows:
        return [
            {"icon": "✅", "text": "Keep practicing with harder questions and add one numerical or comparative example to push the answer into the top band."}
        ]
    return [
        {"icon": "📖", "text": f"Revise <strong>{c['name']}</strong> and rewrite this answer with clearer concept coverage."}
        for c in weak_rows
    ]


def _sample_answer(question: dict) -> str:
    answers = sorted(question["answers"], key=lambda a: float(a["score"]), reverse=True)
    return answers[0]["text"] if answers else ""


def build_question_catalog() -> list[dict]:
    catalog = []
    for q in DATASET:
        catalog.append(
            {
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
                "sample_answer": _sample_answer(q),
                "distribution": _question_distribution(q),
            }
        )
    return catalog


QUESTION_CATALOG = build_question_catalog()


class IntelliPlaceHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/intelliplace_evaluation_engine.html"):
            self._send_html(FRONTEND_PATH.read_text(encoding="utf-8"))
            return
        if path == "/api/questions":
            self._send_json({"questions": QUESTION_CATALOG, "meta_model_loaded": META_ARTIFACT is not None})
            return
        self.send_error(404, "Not Found")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/evaluate":
            self.send_error(404, "Not Found")
            return

        content_len = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_len)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send_json({"error": "Invalid JSON body."}, status=400)
            return

        question_id = payload.get("question_id")
        answer = (payload.get("answer") or "").strip()
        if not question_id or question_id not in QUESTIONS_BY_ID:
            self._send_json({"error": "Unknown or missing question_id."}, status=400)
            return
        if len(answer) < 40:
            self._send_json({"error": "Answer is too short for evaluation."}, status=400)
            return

        q = QUESTIONS_BY_ID[question_id]
        row = _feature_row(q, answer)
        concept_rows = _concept_rows(row["concept_details"], q["rubric"])
        strong, weak = _feedback(concept_rows)
        recs = _recommendations(concept_rows)
        engine_score = round(float(row["raw_score_10"]), 2)

        mode = payload.get("mode", "engine")
        final_score = None
        if mode == "meta":
            final_score = _predict_meta_score(row)
        if final_score is None:
            final_score = engine_score

        demo_reference_score, demo_profile = resolve_intro_demo_final_score(q, answer)
        demo_bracket_applied = demo_reference_score is not None
        demo_score_capped = False
        if demo_bracket_applied:
            raw_ds = float(demo_reference_score)
            strong_cap = float(os.environ.get("INTELLIPLACE_DEMO_STRONG_CAP", "8.5"))
            if demo_profile == "strong":
                final_score = min(raw_ds, strong_cap)
                demo_score_capped = final_score < raw_ds - 1e-6
            else:
                final_score = raw_ds

        # Global contradiction cap REMOVED: NLI contradiction is unreliable
        # for technical CS text. Per-concept handling in concept_detection.py is sufficient.

        verdict, summary = _verdict(final_score)

        if demo_bracket_applied:
            cap_note = ""
            if demo_score_capped and demo_reference_score is not None:
                cap_note = (
                    f" Dataset human label was {float(demo_reference_score):.1f}; "
                    f"display capped (INTELLIPLACE_DEMO_STRONG_CAP)."
                )
            meta_remark = (
                "Intro-question demo: matched a dataset reference row; "
                f"final score for display is {final_score:.1f}/10.{cap_note}"
            )
        elif mode == "meta" and META_ARTIFACT:
            meta_remark = "Meta-model score used for final display."
        else:
            meta_remark = "Meta-model not used; using engine score only."

        remarks = (
            f"Engine score: {engine_score:.2f}/10. {meta_remark} "
            f"The answer was evaluated against {len(q['rubric'])} rubric concepts."
        )
        if demo_bracket_applied:
            remarks += (
                f" Matched profile: {demo_profile} (turn off with INTELLIPLACE_DISABLE_DEMO_BRACKET=1)."
            )
        elif question_id in DEMO_QUESTION_IDS and demo_bracket_enabled():
            remarks += (
                " Tip: copy an answer's text verbatim from this question's answers list in "
                "intelliplace_dataset.json; the intro-question demo maps it to that row's score."
            )

        response = {
            "question_id": question_id,
            "question": q["question"],
            "engine_score": engine_score,
            "meta_score": round(float(final_score), 2),
            "final_score": round(float(final_score), 2),
            "demo_bracket": demo_profile,
            "demo_dataset_reference_score": (
                round(float(demo_reference_score), 2) if demo_bracket_applied and demo_reference_score is not None else None
            ),
            "demo_score_capped": demo_score_capped if demo_bracket_applied else False,
            "verdict": verdict,
            "summary": summary,
            "overall_narrative": (
                f"The evaluation combines concept coverage, contradiction checks, off-topic gating, "
                f"and controlled structural scoring. This answer achieved {round(float(final_score),2)}/10."
            ),
            "badges": [
                {"label": f"Concept {row['concept_score']:.1f}/100", "type": "info"},
                {"label": f"Semantic gate {row['off_topic_gate']:.2f}", "type": "success" if row["off_topic_gate"] >= 0.95 else "warning"},
                {"label": f"Structure {row['structural_score']:.1f}/100", "type": "info"},
                {"label": f"Meta {'on' if mode == 'meta' and META_ARTIFACT else 'off'}", "type": "success" if mode == 'meta' and META_ARTIFACT else "warning"},
            ]
            + (
                [
                    {
                        "label": f"Intro Q demo ({demo_profile})",
                        "type": (
                            "success"
                            if demo_profile == "strong"
                            else "warning"
                            if demo_profile == "weak"
                            else "info"
                        ),
                    }
                ]
                if demo_bracket_applied and demo_profile
                else []
            ),
            "concepts": concept_rows,
            "strong": strong,
            "weak": weak,
            "remarks": remarks,
            "recommendations": recs,
            "mastery": [
                {"label": c["name"], "val": c["score"], "color": "#639922" if c["score"] >= 75 else "#534AB7" if c["score"] >= 45 else "#D85A30"}
                for c in concept_rows
            ],
            "distribution": _question_distribution(q),
        }
        self._send_json(response)


def main() -> None:
    host = "127.0.0.1"
    port = 8000
    server = HTTPServer((host, port), IntelliPlaceHandler)
    print(f"IntelliPlace demo server running at http://{host}:{port}")
    print(f"Meta-model loaded: {META_ARTIFACT is not None}")
    print(f"Platform: {platform.platform()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
