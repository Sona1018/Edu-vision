import os
import sys
from fastapi.testclient import TestClient

# Import the FastAPI application
from backend.app.main import app

def run_e2e_tests():
    print("=================================================================")
    print("🚀 INTELLIPLACE EVALUATION SYSTEM - END-TO-END INTEGRATION TEST")
    print("=================================================================")

    client = TestClient(app)

    # 1. Health check
    print("\n[1/6] Testing Health Check & System Status...")
    res = client.get("/api/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    print("  ✓ Health check passed:", res.json())

    # 2. Pipeline 1: Student Answer Sheet Upload (OCR)
    print("\n[2/6] Testing Pipeline 1 (Student Answer Sheet OCR)...")
    student_pdf_path = os.path.join("storage", "sample_student_sheet.pdf")
    assert os.path.exists(student_pdf_path), "Student PDF not found!"

    with open(student_pdf_path, "rb") as f:
        res = client.post(
            "/api/upload/student",
            files={"file": ("sample_student_sheet.pdf", f, "application/pdf")}
        )

    assert res.status_code == 200, f"Student upload failed: {res.text}"
    student_data = res.json()
    print("  ✓ Pipeline 1 Success!")
    print(f"    - Student File ID: {student_data['file_id']}")
    print(f"    - Pages Rendered: {len(student_data['pages'])}")
    print(f"    - Page 1 Image URL: {student_data['pages'][0]['image_url']}")
    print(f"    - Detected Questions: {list(student_data.get('extracted_answers', {}).keys())}")

    # 3. Pipeline 2: Reference Answer Key Upload & Rubric Parsing
    print("\n[3/6] Testing Pipeline 2 (Reference Answer Key & Rubric Processing)...")
    ref_txt_path = os.path.join("storage", "sample_reference_key.txt")
    assert os.path.exists(ref_txt_path), "Reference text file not found!"

    with open(ref_txt_path, "rb") as f:
        res = client.post(
            "/api/upload/reference",
            files={"file": ("sample_reference_key.txt", f, "text/plain")}
        )

    assert res.status_code == 200, f"Reference upload failed: {res.text}"
    ref_data = res.json()
    print("  ✓ Pipeline 2 Success!")
    print(f"    - Reference File ID: {ref_data['file_id']}")
    print(f"    - Total Questions Parsed: {ref_data['total_questions']}")
    print(f"    - Total Max Marks: {ref_data['total_max_marks']}")
    for q in ref_data.get("questions", []):
        print(f"      * {q['question_id']}: {q['question_text'][:45]}... (Rubric concepts: {len(q.get('rubric', []))})")

    # 4. Central Evaluation Engine
    print("\n[4/6] Testing Central Evaluation Engine (SBERT + DeBERTa + Rubrics)...")
    eval_payload = {
        "student_file_id": student_data["file_id"],
        "reference_file_id": ref_data["file_id"],
        "extracted_answers": student_data.get("extracted_answers") or student_data.get("questions_parsed", {}),
        "qa_bank": ref_data.get("questions", [])
    }

    res = client.post("/api/evaluate", json=eval_payload)
    assert res.status_code == 200, f"Evaluation failed: {res.text}"
    eval_result = res.json()
    eval_id = eval_result["evaluation_id"]

    print("  ✓ Evaluation Engine Success!")
    print(f"    - Evaluation ID: {eval_id}")
    print(f"    - Total Marks Obtained: {eval_result['total_marks_obtained']} / {eval_result['total_max_marks']}")
    print(f"    - Percentage: {eval_result['overall_percentage']}%")
    print(f"    - Overall Grade: {eval_result['overall_grade']}")
    print(f"    - Manual Review Flags: {eval_result['questions_requiring_manual_review']}")

    for q in eval_result["questions"]:
        print(f"\n    --- Question {q['question_id']} Result ---")
        print(f"    * Score: {q['marks_awarded']} / {q['max_marks']} ({q['verdict']})")
        print(f"    * Explanation: {q['explanation'][:90]}...")
        if q.get("missing_key_concepts"):
            print(f"    * Missing Concepts: {q['missing_key_concepts']}")
        if q.get("suggestions_for_improvement"):
            print(f"    * Suggestions: {q['suggestions_for_improvement']}")

    # 5. Testing Manual Review & Score Override
    print("\n[5/6] Testing Evaluator Manual Review & Mark Override...")
    override_payload = {
        "question_number": 1,
        "awarded_marks": 9.5,
        "remarks": "Excellent structured definition and clear marketing mix concepts.",
        "updated_answer": student_data["extracted_answers"].get("Q1", "") + " (Verified by Evaluator)"
    }
    res = client.put(f"/api/results/{eval_id}/override", json=override_payload)
    assert res.status_code == 200, f"Override failed: {res.text}"
    updated_eval = res.json()
    print("  ✓ Manual Override Success!")
    print(f"    - Recalculated Total Marks: {updated_eval['total_marks_obtained']} / {updated_eval['total_max_marks']}")
    print(f"    - Recalculated Percentage: {updated_eval['overall_percentage']}%")
    print(f"    - Recalculated Grade: {updated_eval['overall_grade']}")

    # 6. Testing Export Reports (PDF and Excel)
    print("\n[6/6] Testing Report Generation & Exports...")
    # PDF
    res_pdf = client.get(f"/api/export/{eval_id}/pdf")
    assert res_pdf.status_code == 200, f"PDF export failed: {res_pdf.text}"
    print(f"  ✓ PDF Export Success! Received {len(res_pdf.content)} bytes.")

    # Excel
    res_excel = client.get(f"/api/export/{eval_id}/excel")
    assert res_excel.status_code == 200, f"Excel export failed: {res_excel.text}"
    print(f"  ✓ Excel Export Success! Received {len(res_excel.content)} bytes.")

    print("\n=================================================================")
    print("✅ ALL 6 END-TO-END INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=================================================================")

if __name__ == "__main__":
    run_e2e_tests()
