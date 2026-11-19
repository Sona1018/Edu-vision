# IntelliPlace: AI-Powered Answer Sheet Evaluation System (v2.0)

Automated full-stack evaluation platform for handwritten student answer sheets using OCR, SBERT semantic embedding analysis, DeBERTa NLI contradiction gating, and rubric matching.

---

## 🏛 System Architecture

The application implements a decoupled, dual-pipeline architecture:

```
┌─────────────────────────────────┐       ┌──────────────────────────────────┐
│  PIPELINE 1: Student Sheet OCR   │       │ PIPELINE 2: Reference Answer Key │
│  (PDF / PNG / JPG)              │       │ (PDF / TXT / DOCX)               │
└────────────────┬────────────────┘       └────────────────┬─────────────────┘
                 │                                         │
                 ▼                                         ▼
      PyMuPDF & Tesseract OCR                   QA Bank & Rubric Parser
   - Page-by-page Image Render               - Concept Weight Normalization
   - OCR Noise Cleaning & Text Extraction    - Model Answers & Marking Schemes
   - Question Boundary Splitting             - Max Marks Criteria
                 │                                         │
                 └───────────────────┬─────────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │     CENTRAL EVALUATION ENGINE     │
                   │  - SBERT (all-mpnet-base-v2)      │
                   │  - DeBERTa NLI (Contradictions)   │
                   │  - Rubric Concept Matching        │
                   │  - Structural Depth & Word Count  │
                   │  - Review Flagging & Suggestions  │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
             ┌───────────────────────────────────────────────┐
             │       MONOCHROMATIC RESULTS DASHBOARD         │
             │  • Overall Scorecard & Performance Metrics    │
             │  • Question-wise Expandable Assessment Cards  │
             │  • Side-by-Side Handwriting vs OCR Viewer     │
             │  • Evaluator Manual Review & Mark Overrides   │
             │  • One-Click PDF & Excel Report Exports       │
             └───────────────────────────────────────────────┘
```

---

## 🎨 Design Philosophy
- **Monochromatic Minimalist SaaS Aesthetic**: Built strictly with a black and white / charcoal palette (`#09090b` zinc-950 background, `#121214` dark charcoal cards, subtle `#27272a` borders, and high-contrast white text and buttons).
- Zero distracting gradients, blues, or purple tints.
- High-density typography, professional spacing, responsive layouts.

---

## 🚀 Quick Start Guide

### 1. Requirements
- Python 3.10+
- Node.js 18+ (for frontend development)

### 2. Run the Full-Stack Application
Start the integrated server with a single command:
```bash
python run_app.py
```
Open your browser at:
- **Web Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 Automated End-to-End Tests
Run the comprehensive test suite verifying both upload pipelines, the AI evaluation engine, manual overrides, and PDF/Excel generation:
```bash
python test_e2e_pipeline.py
```

All 6 integration tests will execute and confirm:
1. Health check & status
2. Pipeline 1 OCR & Page Image Rendering
3. Pipeline 2 Reference Key & Rubric Parsing
4. Central Evaluation Engine (SBERT + DeBERTa)
5. Manual Review Override & Dynamic Score Recalculation
6. PDF & Excel Report Exports

---

## 📂 Project Structure

```
IntelliPlace-AI-Answer-Grading-System-main/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI app, static mounts, CORS, routes
│   │   ├── routes/
│   │   │   ├── student.py              # POST /api/upload/student
│   │   │   ├── reference.py            # POST /api/upload/reference
│   │   │   └── evaluation.py           # POST /api/evaluate, overrides, exports
│   │   ├── services/
│   │   │   ├── ocr_service.py          # Pipeline 1: PyMuPDF + Tesseract OCR
│   │   │   ├── reference_service.py    # Pipeline 2: QA Bank & Rubric Parser
│   │   │   └── evaluation_service.py   # SBERT + DeBERTa + Rubrics Engine
│   │   └── utils/
│   │       ├── file_handler.py         # File validation & UUID storage
│   │       └── report_generator.py     # PDF (ReportLab) & Excel (openpyxl)
│   └── requirements.txt
├── frontend/
│   └── ui/
│       ├── src/
│       │   ├── App.jsx                 # Monochromatic React dashboard
│       │   ├── main.jsx
│       │   └── index.css               # Tailwind CSS styles
│       ├── dist/                       # Pre-built production client
│       ├── package.json
│       └── vite.config.js
├── storage/
│   ├── uploads/                        # Saved upload files
│   ├── page_images/                    # High-res rendered page images
│   └── reports/                        # Generated PDF & Excel files
├── run_app.py                          # Single-command launcher
└── test_e2e_pipeline.py                # End-to-end integration test suite
```

---

## 🔍 Key Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload/student` | Upload student answer sheet (PDF/PNG/JPG). Returns page images and extracted answers. |
| `POST` | `/api/upload/reference` | Upload reference answer key (TXT/PDF/DOCX). Returns structured QA bank with rubrics. |
| `POST` | `/api/evaluate` | Evaluates student answers against reference key using SBERT + DeBERTa. |
| `GET`  | `/api/results/{id}` | Retrieves full evaluation report. |
| `PUT`  | `/api/results/{id}/override` | Manually updates marks, edits answers, adds teacher remarks, and recalculates score. |
| `GET`  | `/api/export/{id}/pdf` | Generates and downloads evaluation report as PDF. |
| `GET`  | `/api/export/{id}/excel` | Generates and downloads evaluation report as Excel sheet. |
| `GET`  | `/api/health` | System health check. |
