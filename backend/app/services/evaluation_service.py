import os
import re
import uuid
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
# Check device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OFFLINE_MODE = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"

# Global model placeholders
_bi_encoder = None
_nli_tokenizer = None
_nli_model = None

def get_models():
    global _bi_encoder, _nli_tokenizer, _nli_model
    if _bi_encoder is None:
        print(f"🧠 Loading SentenceTransformer on {DEVICE}...")
        from sentence_transformers import SentenceTransformer
        _bi_encoder = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)
    if _nli_tokenizer is None or _nli_model is None:
        print(f"🧠 Loading DeBERTa NLI on {DEVICE}...")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        nli_name = "cross-encoder/nli-deberta-v3-small"
        _nli_tokenizer = AutoTokenizer.from_pretrained(nli_name)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(
            nli_name,
            low_cpu_mem_usage=True
        ).to(DEVICE)
        _nli_model.eval()
    return _bi_encoder, _nli_tokenizer, _nli_model

# In-memory storage for evaluations
EVALUATIONS_DB: Dict[str, Any] = {}

def get_nli_probabilities(premise: str, hypothesis: str, tokenizer, model) -> tuple:
    """Returns (entailment_prob, contradiction_prob)."""
    try:
        inputs = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)[0]
        # label order: 0=entailment, 1=neutral, 2=contradiction
        return float(probs[0].item()), float(probs[2].item())
    except Exception:
        return 0.5, 0.0

def evaluate_single_question(
    q_data: dict,
    student_answer: str,
    bi_encoder,
    nli_tokenizer,
    nli_model
) -> dict:
    """
    Evaluates one question using SBERT + DeBERTa NLI + Rubric Scoring.
    """
    q_id = q_data.get("question_id", "Q1")
    q_text = q_data.get("question_text", "")
    ref_answer = q_data.get("reference_answer", "")
    max_marks = float(q_data.get("max_marks", 10.0))
    rubric = q_data.get("rubric", [])
    
    if not student_answer or len(student_answer.strip()) < 15:
        return {
            "question_id": q_id,
            "question_text": q_text,
            "reference_answer": ref_answer,
            "extracted_student_answer": student_answer or "[No answer detected]",
            "max_marks": max_marks,
            "marks_awarded": 0.0,
            "percentage": 0.0,
            "verdict": "Unattempted / Empty",
            "status_badge": "incorrect",
            "explanation": "No substantial answer was detected for this question in the student sheet.",
            "missing_key_concepts": [c.get("concept_name", "") for c in rubric],
            "incorrect_statements": [],
            "suggestions_for_improvement": ["Review the topic and provide a complete structured explanation."],
            "concept_matches": [],
            "needs_manual_review": True,
            "confidence": "Low"
        }
        
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', student_answer) if len(s.strip()) > 8]
    if not sentences:
        sentences = [student_answer.strip()]
        
    sent_embs = bi_encoder.encode(sentences, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    student_doc_emb = bi_encoder.encode(student_answer, convert_to_numpy=True)
    ideal_doc_emb = bi_encoder.encode(ref_answer, convert_to_numpy=True)
    
    # 1. Holistic Semantic Score & Off-Topic Gating
    raw_semantic = float(cosine_similarity([student_doc_emb], [ideal_doc_emb])[0][0])
    raw_semantic = max(0.0, min(1.0, raw_semantic))
    
    off_topic_gate = 1.0 if raw_semantic >= 0.35 else max(0.48, raw_semantic / 0.35)
    
    # 2. Rubric Concept Matching
    total_rubric_weight = sum(float(c.get("weight", 1.0)) for c in rubric) or 1.0
    weighted_concept_score = 0.0
    concept_matches = []
    missing_concepts = []
    incorrect_statements = []
    
    for c in rubric:
        c_name = c.get("concept_name", "")
        c_ref = c.get("reference_explanation", c_name)
        weight = float(c.get("weight", 1.0))
        
        c_emb = bi_encoder.encode(c_ref, convert_to_numpy=True)
        sims = cosine_similarity(sent_embs, [c_emb]).flatten()
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_sentence = sentences[best_idx]
        
        doc_sim = float(cosine_similarity([student_doc_emb], [c_emb])[0][0])
        blended_sim = 0.80 * best_sim + 0.20 * doc_sim
        
        # NLI Contradiction & Entailment
        ent, con = get_nli_probabilities(best_sentence, c_ref, nli_tokenizer, nli_model)
        
        adjusted_sim = blended_sim
        if con > 0.65 and con > ent:
            adjusted_sim *= 0.68
            incorrect_statements.append(f"In '{best_sentence[:100]}...': Contradicts reference key regarding '{c_name}'.")
        elif con > 0.50 and con > ent + 0.08:
            adjusted_sim *= 0.78
            
        adjusted_sim = float(np.clip(adjusted_sim, 0.0, 1.0))
        earned_concept_score = weight * adjusted_sim
        weighted_concept_score += earned_concept_score
        
        match_pct = round(adjusted_sim * 100, 1)
        concept_matches.append({
            "concept_name": c_name,
            "weight_pct": f"{int(round((weight / total_rubric_weight) * 100))}%",
            "match_pct": f"{match_pct}%",
            "best_sentence": best_sentence[:120],
            "status": "full" if match_pct >= 70 else ("partial" if match_pct >= 35 else "missed")
        })
        
        if match_pct < 40:
            missing_concepts.append(c_name)
            
    norm_concept_score = (weighted_concept_score / total_rubric_weight) * off_topic_gate
    
    # 3. Structural Depth Score
    word_count = len(student_answer.split())
    if word_count < 25:
        struct_factor = 0.40
    elif word_count < 60:
        struct_factor = 0.65
    elif word_count < 120:
        struct_factor = 0.85
    else:
        struct_factor = 1.0
        
    # Hybrid combination (78% concept, 13% semantic, 9% structure)
    hybrid_ratio = (0.78 * norm_concept_score) + (0.13 * raw_semantic) + (0.09 * struct_factor)
    hybrid_ratio = float(np.clip(hybrid_ratio, 0.0, 1.0))
    
    # Final Marks Scaled to Max Marks
    marks_awarded = round(hybrid_ratio * max_marks, 2)
    score_pct = round((marks_awarded / max_marks) * 100, 1)
    
    # Verdict & Badge
    if score_pct >= 80:
        verdict = "Excellent / Complete"
        status_badge = "correct"
        explanation = f"Comprehensive answer covering {len(rubric) - len(missing_concepts)} of {len(rubric)} key concepts with strong clarity and examples."
    elif score_pct >= 60:
        verdict = "Good Understanding"
        status_badge = "partially_correct"
        explanation = f"Good grasp of fundamental concepts, but could benefit from deeper technical detail on {missing_concepts[0] if missing_concepts else 'sub-points'}."
    elif score_pct >= 40:
        verdict = "Partial Credit"
        status_badge = "partially_correct"
        explanation = "Basic understanding demonstrated with incomplete rubric coverage or brevity."
    else:
        verdict = "Needs Improvement"
        status_badge = "incorrect"
        explanation = f"Answer is either brief or misses critical concepts: {', '.join(missing_concepts[:2])}."
        
    # Suggestions for improvement
    suggestions = []
    if missing_concepts:
        suggestions.append(f"Include explanations for: {', '.join(missing_concepts[:3])}.")
    if word_count < 75:
        suggestions.append("Elaborate further with real-world industry examples or structured bullet points.")
    if incorrect_statements:
        suggestions.append("Clarify contradictory technical points noted in the explanation.")
    if not suggestions:
        suggestions.append("Solid answer. Maintain this level of detail and formatting!")
        
    needs_review = (score_pct < 45 and word_count > 100) or len(incorrect_statements) > 0
    confidence = "High" if len(sentences) >= 3 and raw_semantic > 0.40 else "Medium"
    
    return {
        "question_id": q_id,
        "question_text": q_text,
        "reference_answer": ref_answer,
        "extracted_student_answer": student_answer,
        "max_marks": max_marks,
        "marks_awarded": marks_awarded,
        "percentage": score_pct,
        "verdict": verdict,
        "status_badge": status_badge,
        "explanation": explanation,
        "missing_key_concepts": missing_concepts,
        "incorrect_statements": incorrect_statements,
        "suggestions_for_improvement": suggestions,
        "concept_matches": concept_matches,
        "needs_manual_review": needs_review,
        "confidence": confidence
    }

def run_evaluation_pipeline(
    student_data: dict,
    reference_data: dict
) -> dict:
    """
    Combines Pipeline 1 (Student OCR data) and Pipeline 2 (Reference QA Bank)
    into a complete Question-wise Evaluation Report.
    """
    evaluation_id = str(uuid.uuid4())
    bi_encoder, nli_tokenizer, nli_model = get_models()
    
    student_questions = student_data.get("questions_parsed", {})
    ref_questions = reference_data.get("questions", [])
    
    evaluated_questions = []
    total_obtained = 0.0
    total_max = 0.0
    manual_review_count = 0
    
    # Question-wise evaluation loop
    for q_item in ref_questions:
        q_id = q_item.get("question_id")
        # Try matching question key
        stu_ans = student_questions.get(q_id, "")
        if not stu_ans:
            # Try numeric match like "1" instead of "Q1"
            q_num = str(q_item.get("question_number", ""))
            stu_ans = student_questions.get(f"Q{q_num}", student_questions.get(q_num, ""))
            
        result = evaluate_single_question(q_item, stu_ans, bi_encoder, nli_tokenizer, nli_model)
        evaluated_questions.append(result)
        
        total_obtained += result["marks_awarded"]
        total_max += result["max_marks"]
        if result["needs_manual_review"]:
            manual_review_count += 1
            
    overall_pct = round((total_obtained / total_max) * 100, 1) if total_max > 0 else 0.0
    
    # Build complete evaluation report object
    report = {
        "evaluation_id": evaluation_id,
        "timestamp": str(uuid.uuid1()),
        "student_file_id": student_data.get("file_id"),
        "reference_file_id": reference_data.get("file_id"),
        "total_marks_obtained": round(total_obtained, 2),
        "total_max_marks": round(total_max, 2),
        "overall_percentage": overall_pct,
        "total_questions_evaluated": len(evaluated_questions),
        "questions_requiring_manual_review": manual_review_count,
        "overall_grade": "A+" if overall_pct >= 85 else ("A" if overall_pct >= 75 else ("B" if overall_pct >= 60 else ("C" if overall_pct >= 45 else "D"))),
        "questions": evaluated_questions,
        # Summary object for report generator & dashboard cards
        "summary": {
            "total_obtained": round(total_obtained, 2),
            "total_max": round(total_max, 2),
            "percentage": overall_pct,
            "total_questions": len(evaluated_questions),
            "review_flags_count": manual_review_count,
            "grade": "A+" if overall_pct >= 85 else ("A" if overall_pct >= 75 else ("B" if overall_pct >= 60 else ("C" if overall_pct >= 45 else "D")))
        },
        "question_evaluations": evaluated_questions,
        "pages": student_data.get("pages", []),
        "student_metadata": {
            "total_pages": student_data.get("total_pages", len(student_data.get("pages", []))),
            "total_words": student_data.get("total_words", 0),
            "ocr_confidence": student_data.get("confidence", "Medium")
        }
    }
    
    EVALUATIONS_DB[evaluation_id] = report
    return report

def run_full_evaluation(
    extracted_answers: dict,
    qa_bank: dict,
    student_file_id: str = None,
    reference_file_id: str = None,
    pages: list = None
) -> dict:
    """Convenience wrapper for evaluation API endpoint."""
    student_data = {
        "file_id": student_file_id,
        "questions_parsed": extracted_answers,
        "pages": pages or []
    }
    # qa_bank can be a list or a dict
    if isinstance(qa_bank, dict):
        if "questions" in qa_bank:
            ref_questions = qa_bank["questions"]
        else:
            # list of values or dict mapping
            ref_questions = list(qa_bank.values())
    elif isinstance(qa_bank, list):
        ref_questions = qa_bank
    else:
        ref_questions = []
        
    reference_data = {
        "file_id": reference_file_id,
        "questions": ref_questions
    }
    return run_evaluation_pipeline(student_data, reference_data)

def get_evaluation(evaluation_id: str) -> dict:
    return EVALUATIONS_DB.get(evaluation_id)

def get_all_evaluations() -> list:
    return list(EVALUATIONS_DB.values())

def update_manual_override(
    evaluation_id: str,
    question_number: int = None,
    awarded_marks: float = None,
    remarks: str = "",
    updated_answer: str = None,
    overrides: dict = None
) -> dict:
    """
    Applies manual teacher edits/overrides to marks, text, or remarks
    and automatically recalculates the total scores.
    """
    if evaluation_id not in EVALUATIONS_DB:
        return None
        
    report = EVALUATIONS_DB[evaluation_id]
    
    if overrides is None:
        overrides = {}
        if question_number is not None:
            overrides[f"Q{question_number}"] = {
                "marks_awarded": awarded_marks,
                "teacher_remarks": remarks,
                "edited_student_answer": updated_answer
            }
            overrides[str(question_number)] = overrides[f"Q{question_number}"]
    
    total_obtained = 0.0
    manual_review_count = 0
    
    for q in report["questions"]:
        qid = q.get("question_id", "")
        # match Q1, 1, or question_number
        matched_ov = None
        for k in [qid, qid.replace("Q", ""), f"Q{q.get('question_number', '')}", str(q.get('question_number', ''))]:
            if k in overrides:
                matched_ov = overrides[k]
                break
                
        if matched_ov:
            if matched_ov.get("marks_awarded") is not None:
                new_marks = float(matched_ov["marks_awarded"])
                q["marks_awarded"] = round(min(q["max_marks"], max(0.0, new_marks)), 2)
                q["awarded_marks"] = q["marks_awarded"]
                q["percentage"] = round((q["marks_awarded"] / q["max_marks"]) * 100, 1)
                q["verdict"] = "Manually Verified / Overridden"
                q["status_badge"] = "correct" if q["percentage"] >= 65 else "partially_correct"
                q["needs_manual_review"] = False
            if matched_ov.get("teacher_remarks"):
                q["teacher_remarks"] = matched_ov["teacher_remarks"]
                q["remarks"] = matched_ov["teacher_remarks"]
            if matched_ov.get("edited_student_answer"):
                q["extracted_student_answer"] = matched_ov["edited_student_answer"]
                q["student_answer"] = matched_ov["edited_student_answer"]
                
        total_obtained += q["marks_awarded"]
        if q.get("needs_manual_review"):
            manual_review_count += 1
        
    overall_pct = round((total_obtained / report["total_max_marks"]) * 100, 1) if report["total_max_marks"] > 0 else 0.0
    report["total_marks_obtained"] = round(total_obtained, 2)
    report["overall_percentage"] = overall_pct
    report["questions_requiring_manual_review"] = manual_review_count
    report["overall_grade"] = "A+" if overall_pct >= 85 else ("A" if overall_pct >= 75 else ("B" if overall_pct >= 60 else ("C" if overall_pct >= 45 else "D")))
    
    report["summary"]["total_obtained"] = report["total_marks_obtained"]
    report["summary"]["percentage"] = overall_pct
    report["summary"]["review_flags_count"] = manual_review_count
    report["summary"]["grade"] = report["overall_grade"]
    
    EVALUATIONS_DB[evaluation_id] = report
    return report

def run_full_evaluation(
    extracted_answers: dict,
    qa_bank: dict,
    student_file_id: str = None,
    reference_file_id: str = None,
    pages: list = None
) -> dict:
    """Convenience wrapper for evaluation API endpoint."""
    student_data = {
        "file_id": student_file_id,
        "questions_parsed": extracted_answers,
        "pages": pages or []
    }
    # qa_bank can be a list or a dict
    if isinstance(qa_bank, dict):
        if "questions" in qa_bank:
            ref_questions = qa_bank["questions"]
        else:
            # list of values or dict mapping
            ref_questions = list(qa_bank.values())
    elif isinstance(qa_bank, list):
        ref_questions = qa_bank
    else:
        ref_questions = []
        
    reference_data = {
        "file_id": reference_file_id,
        "questions": ref_questions
    }
    return run_evaluation_pipeline(student_data, reference_data)
