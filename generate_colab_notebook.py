import json

def create_notebook():
    cells = []

    def add_md(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def add_code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # Header Markdown
    add_md("""# 🎓 IntelliPlace — AI Answer Grading System (Google Colab)

This notebook evaluates descriptive technical student answers using the **IntelliPlace AI Architecture**:
1. **Semantic Similarity** using SBERT (`all-mpnet-base-v2`)
2. **Rubric-based Concept Scoring** with top-k evidence matching & hinge remapping
3. **NLI Contradiction / Hallucination Detection** using DeBERTa (`cross-encoder/nli-deberta-v3-small`)
4. **Structural Completeness & Mastery Analysis**
5. **Hierarchical 24-Page Long Document Handling** (handles large OCR texts / multi-page PDF booklets without token-limit truncation)

---
### 📥 3 Dedicated Input Cells:
- 📌 **INPUT 1:** Reference Question (Exam / Interview Question)
- 📌 **INPUT 2:** Reference Answer (Ideal Model Answer or Rubric Points)
- 📌 **INPUT 3:** Student Answer (Pasted OCR text OR Direct 24-page PDF booklet upload)

---
💡 **Tip:** On Colab, enable GPU acceleration via **Runtime > Change runtime type > T4 GPU** for 10x faster execution.""")

    # Cell 1: Install packages
    add_code("""#@title ⚙️ Step 1: Install Dependencies
!pip install -q --upgrade pillow
!pip install -q sentence-transformers transformers scikit-learn pypdf pdfplumber tabulate

print("✅ Dependencies installed successfully!")
print("💡 Tip: If you get a 'Restart session' button, click it once, then run the remaining cells.")""")

    # Cell 2: Imports & Device
    add_code("""#@title 🚀 Step 2: Load Libraries & Verify GPU
import os
import re
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tabulate import tabulate
from IPython.display import display, HTML

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running on Device: {device}")
if torch.cuda.is_available():
    print(f"🎮 GPU Model: {torch.cuda.get_device_name(0)}")
else:
    print("ℹ️ Running on CPU. (Tip: Switch to T4 GPU in Runtime settings for 10x faster inference)")""")

    # Cell 3: Load Models
    add_code("""#@title 🧠 Step 3: Load AI Models (SBERT + DeBERTa NLI)
print("⏳ Loading SBERT bi-encoder (all-mpnet-base-v2)...")
bi_encoder = SentenceTransformer("all-mpnet-base-v2", device=device)

print("⏳ Loading NLI cross-encoder (cross-encoder/nli-deberta-v3-small)...")
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
nli_model = AutoModelForSequenceClassification.from_pretrained(
    NLI_MODEL_NAME,
    low_cpu_mem_usage=True
).to(device)
nli_model.eval()

print("✅ Both AI models loaded successfully into memory!")""")

    # Divider Markdown for Inputs
    add_md("""---
## 📝 ENTER YOUR 3 INPUTS BELOW
Fill in **Input 1**, **Input 2**, and **Input 3** in the respective cells below, then run the final **🚀 RUN EVALUATION** cell!""")

    # Cell 4: Input 1
    add_code("""#@title 📌 INPUT 1: Reference Question
REFERENCE_QUESTION = "What do you mean by an Operating System? What are its core functions?" #@param {type:"string"}

print("✅ Saved Reference Question:")
print(f"👉 {REFERENCE_QUESTION}")""")

    # Cell 5: Input 2
    add_code("""#@title 📌 INPUT 2: Reference Answer / Model Answer
REFERENCE_ANSWER = \"\"\"
1. Definition of Operating System: An OS is a system software that acts as an interface between computer hardware and user applications, managing all system resources.
2. Resource Management: The OS coordinates CPU scheduling, main memory allocation, secondary storage, files, and I/O devices to ensure conflict-free sharing.
3. Core OS Functions: Key functions include processor management, memory management, file system organization, device management, error detection, security protection, and user interface provision.
\"\"\" #@param {type:"string"}

print("✅ Saved Reference Answer:")
print(REFERENCE_ANSWER.strip())""")

    # Cell 6: Input 3
    add_code("""#@title 📌 INPUT 3: Student Answer (OCR Text OR 24-Page PDF Booklet)
import re
import os

def clean_ocr_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    text = re.sub(r'(\\w+)-\\s*\\n\\s*(\\w+)', r'\\1\\2', raw_text)
    text = text.replace('\\u00a0', ' ').replace('\\t', ' ')
    text = re.sub(r'\\n\\s*\\n+', '\\n\\n', text)
    text = re.sub(r'(?i)page\\s+\\d+(\\s*(of|/)\\s*\\d+)?', '', text)
    text = re.sub(r'[ ]{2,}', ' ', text)
    return text.strip()

def extract_text_from_pdf(pdf_path: str) -> str:
    extracted_text = []
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        print(f"📄 Detected {len(reader.pages)} pages in PDF.")
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            extracted_text.append(f"\\n--- [Page {i+1}] ---\\n" + page_text)
    except Exception:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                extracted_text.append(f"\\n--- [Page {i+1}] ---\\n" + page_text)
    return clean_ocr_text("\\n".join(extracted_text))

UPLOAD_PDF_NOW = False #@param {type:"boolean"}
PDF_FILE_PATH = "" #@param {type:"string"}

STUDENT_OCR_ANSWER = \"\"\"
An operating system is the master control software that sits directly between the computer hardware and the user applications.
Without an operating system, the computer hardware cannot be easily utilized because user programs would have to interact
directly with complex machine circuits.

Its first primary responsibility is resource management. The OS coordinates the CPU by scheduling which process runs at
what time using scheduling algorithms. It also handles main memory (RAM) management, ensuring processes do not overwrite
each other's address space. It organizes secondary storage through file systems so documents and directories can be stored
and retrieved reliably.

Additionally, the operating system manages all peripheral input/output devices like keyboards, disks, displays, and network
adapters through device drivers. It also offers a user interface whether CLI or GUI, detects system hardware and software
errors, and provides basic security mechanisms like user authentication and file permission controls so unauthorized users
cannot tamper with private data.
\"\"\" #@param {type:"string"}

resolved_student_text = ""

if UPLOAD_PDF_NOW:
    from google.colab import files
    print("📤 Select and upload your 24-page PDF booklet:")
    uploaded = files.upload()
    if uploaded:
        uploaded_pdf_name = list(uploaded.keys())[0]
        print(f"✅ Uploaded PDF: {uploaded_pdf_name}")
        resolved_student_text = extract_text_from_pdf(uploaded_pdf_name)
elif PDF_FILE_PATH.strip() and os.path.exists(PDF_FILE_PATH.strip()):
    print(f"📄 Reading PDF file from path: {PDF_FILE_PATH}")
    resolved_student_text = extract_text_from_pdf(PDF_FILE_PATH.strip())
else:
    print("📝 Using pasted student OCR text.")
    resolved_student_text = clean_ocr_text(STUDENT_OCR_ANSWER)

words_detected = len(resolved_student_text.split())
print(f"\\n✅ Student Answer Ready for Grading!")
print(f"📊 Total Words Detected: {words_detected} words across {len(resolved_student_text)} characters.")""")

    # Cell 7: Run Evaluation (Completely self-contained with auto-model load)
    add_code("""#@title 🚀 RUN EVALUATION & VIEW REPORT
#@markdown Click the **Play** button to evaluate your answer!

import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display, HTML

# -----------------------------------------------------------------------------
# AUTO-LOAD MODELS IF NOT ALREADY LOADED IN MEMORY
# -----------------------------------------------------------------------------
if 'bi_encoder' not in globals() or 'nli_model' not in globals():
    print("⏳ AI models not detected in memory. Automatically loading them now...")
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    print("⏳ Loading SBERT bi-encoder (all-mpnet-base-v2)...")
    bi_encoder = SentenceTransformer("all-mpnet-base-v2", device=device)

    print("⏳ Loading NLI cross-encoder (cross-encoder/nli-deberta-v3-small)...")
    NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL_NAME,
        low_cpu_mem_usage=True
    ).to(device)
    nli_model.eval()
    print("✅ AI Models loaded successfully!")

# Ensure inputs exist
if 'REFERENCE_QUESTION' not in globals():
    REFERENCE_QUESTION = "What do you mean by an Operating System? What are its core functions?"
if 'REFERENCE_ANSWER' not in globals():
    REFERENCE_ANSWER = "1. Definition of OS: Software that manages hardware resources.\\n2. Resource Management: Handles CPU, Memory, Disk, and I/O devices.\\n3. Core Functions: Process scheduling, file systems, security, and UI."
if 'resolved_student_text' not in globals() or not resolved_student_text:
    resolved_student_text = globals().get('STUDENT_OCR_ANSWER', '')

# -----------------------------------------------------------------------------
# ENGINE CONSTANTS & SCORING LOGIC
# -----------------------------------------------------------------------------
OFF_TOPIC_THRESHOLD = 0.35
GATE_MIN = 0.48
GATE_MAX = 1.00
SIMILARITY_BASELINE = 0.12
TOP_K_SENTENCES = 3
TOP_K_WEIGHTS = [1.0, 0.50, 0.20]
DOC_LEVEL_BLEND = 0.20
HINGE_LOW = 0.26
HINGE_MID = 0.50
HINGE_HIGH = 0.72
CONTRADICTION_GATE_THRESH = 0.72
ENTAILMENT_SHRINK_THRESH = 0.66
SBERT_SKIP_ENTAILMENT_SHRINK_IF_BLEND_GE = 0.52
CONCEPT_COVERAGE_BONUS = 0.03
COVERAGE_THRESHOLD = 0.31
PRIMARY_SENTENCE_REUSE_PENALTY = 0.52
WEIGHT_CONCEPT = 0.78
WEIGHT_STRUCT = 0.09
WEIGHT_SEMANTIC = 0.13
STRUCTURAL_CEILING = 78.0
COVERAGE_CAP_60 = 82.0
COVERAGE_CAP_40 = 70.0
COVERAGE_CAP_NONE = 50.0
SOFT_COVERAGE_BAND = 0.11
CONTRADICTION_RATIO_THRESH = 0.62
CONTRADICTION_ADDITIVE_MAX = 12.0

def split_into_sentences(text: str) -> list[str]:
    raw = re.split(r'(?<=[.!?])\\s+', text.strip())
    clean = [s.strip() for s in raw if len(s.strip()) > 8 and len(s.strip().split()) >= 2]
    return clean if clean else ([text.strip()] if text.strip() else [])

def get_hierarchical_doc_embedding(text: str, chunk_words: int = 180, overlap: int = 40) -> np.ndarray:
    words = text.split()
    if len(words) <= chunk_words:
        return bi_encoder.encode(text, convert_to_numpy=True)
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_words])
        chunks.append(chunk)
        start += (chunk_words - overlap)
    print(f"📑 Split 24-page answer ({len(words)} words) into {len(chunks)} sliding-window chunks for complete representation.")
    chunk_embs = bi_encoder.encode(chunks, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    doc_emb = np.mean(chunk_embs, axis=0)
    norm = np.linalg.norm(doc_emb)
    return (doc_emb / norm) if norm > 0 else doc_emb

def build_rubric_from_reference(reference_answer: str, max_concepts: int = 6) -> list[dict]:
    lines = [line.strip() for line in reference_answer.strip().split(\"\\n\") if line.strip()]
    bullet_items = [re.sub(r'^[\\d+\\.\\-*•\\s]+', '', line) for line in lines if len(line) > 15]
    concepts = []
    if len(bullet_items) >= 2:
        items = bullet_items[:max_concepts]
        weight_per_concept = round(1.0 / len(items), 3)
        for i, item in enumerate(items):
            first_words = " ".join(item.split()[:4])
            concepts.append({
                "concept_name": f"Concept {i+1}: {first_words}...",
                "reference_explanation": item,
                "weight": weight_per_concept
            })
    else:
        sents = split_into_sentences(reference_answer)
        items = sents[:max_concepts] if len(sents) > 3 else sents
        weight_per_concept = round(1.0 / len(items), 3)
        for i, s in enumerate(items):
            first_words = " ".join(s.split()[:4])
            concepts.append({
                "concept_name": f"Key Point {i+1}: {first_words}...",
                "reference_explanation": s,
                "weight": weight_per_concept
            })
    tot = sum(c["weight"] for c in concepts)
    if tot > 0:
        for c in concepts:
            c["weight"] = round(c["weight"] / tot, 3)
    return concepts

def _hinge_remap(raw_sim: float) -> float:
    if raw_sim < HINGE_LOW: return 0.0
    elif raw_sim < HINGE_MID: return ((raw_sim - HINGE_LOW) / (HINGE_MID - HINGE_LOW)) * 0.40
    elif raw_sim < HINGE_HIGH: return 0.40 + ((raw_sim - HINGE_MID) / (HINGE_HIGH - HINGE_MID)) * 0.40
    else: return 0.80 + ((raw_sim - HINGE_HIGH) / (1.0 - HINGE_HIGH)) * 0.20

def get_entailment_score(premise: str, hypothesis: str):
    inputs = nli_tokenizer(premise, hypothesis, return_tensors=\"pt\", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = F.softmax(logits, dim=1)[0]
    return float(probs[0].item()), float(probs[2].item())

def compute_semantic_score(student_doc_emb: np.ndarray, ideal_doc_emb: np.ndarray):
    raw_similarity = float(cosine_similarity([student_doc_emb], [ideal_doc_emb])[0][0])
    raw_similarity = max(0.0, raw_similarity)
    corrected = (raw_similarity - SIMILARITY_BASELINE) / (1.0 - SIMILARITY_BASELINE)
    corrected = float(np.clip(corrected, 0.0, 1.0))
    score = round(corrected * 100.0, 2)
    gate_signal = 0.58 * raw_similarity + 0.42 * corrected
    if gate_signal < OFF_TOPIC_THRESHOLD:
        gate_ratio = gate_signal / OFF_TOPIC_THRESHOLD
        off_topic_gate = GATE_MIN + gate_ratio * (GATE_MAX - GATE_MIN)
    else:
        off_topic_gate = GATE_MAX
    return score, float(np.clip(off_topic_gate, GATE_MIN, GATE_MAX))

def compute_concept_score(sentences: list[str], sentence_embeddings: np.ndarray, full_answer_emb: np.ndarray, rubric: list):
    if len(sentences) == 0: return 0.0, []
    total_weight = sum(c.get("weight", 1.0) for c in rubric) or 1.0
    weighted_score = 0.0
    concept_details = []
    concepts_above_threshold = 0
    used_primary_indices = set()
    n_sent = len(sentences)
    n_con = len(rubric)
    slack = max(0.0, float(n_con + 2 - n_sent))
    reuse_penalty_mult = PRIMARY_SENTENCE_REUSE_PENALTY + (1.0 - PRIMARY_SENTENCE_REUSE_PENALTY) * min(1.0, slack / 3.0)

    for concept in rubric:
        ref_text = concept["reference_explanation"]
        weight = float(concept.get("weight", 1.0))
        ref_embedding = bi_encoder.encode(ref_text, convert_to_numpy=True)
        sims_ref = cosine_similarity(sentence_embeddings, [ref_embedding]).flatten()
        raw_similarities = np.clip(sims_ref, 0.0, 1.0)
        pick_scores = raw_similarities.astype(float).copy()
        for idx in used_primary_indices:
            pick_scores[idx] *= reuse_penalty_mult
        best_idx = int(np.argmax(pick_scores))
        max_raw_sim = float(raw_similarities[best_idx])
        best_sentence = sentences[best_idx]
        used_primary_indices.add(best_idx)

        k = min(TOP_K_SENTENCES, len(sentences))
        top_indices = np.argsort(raw_similarities)[-k:][::-1]
        weights_used = TOP_K_WEIGHTS[:k]
        weights_used = [w / sum(weights_used) for w in weights_used]
        top_k_quality = sum(weights_used[rank] * float(raw_similarities[idx]) for rank, idx in enumerate(top_indices))

        doc_sim = float(cosine_similarity([full_answer_emb], [ref_embedding])[0][0])
        doc_sim = float(np.clip(doc_sim, 0.0, 1.0))
        blended_sim = (1.0 - DOC_LEVEL_BLEND) * top_k_quality + DOC_LEVEL_BLEND * doc_sim
        hinge_sim = _hinge_remap(blended_sim)

        entailment_prob, contradiction_prob = get_entailment_score(best_sentence, ref_text)
        adjusted = hinge_sim
        if contradiction_prob > CONTRADICTION_GATE_THRESH and contradiction_prob > entailment_prob:
            adjusted *= 0.66 if doc_sim >= 0.50 else 0.52
        elif contradiction_prob > 0.52 and contradiction_prob > entailment_prob + 0.08:
            adjusted *= 0.72
        if adjusted > 0.0 and entailment_prob < ENTAILMENT_SHRINK_THRESH:
            if blended_sim < SBERT_SKIP_ENTAILMENT_SHRINK_IF_BLEND_GE:
                adjusted *= 0.82
        adjusted = float(np.clip(adjusted, 0.0, 1.0))
        if adjusted >= COVERAGE_THRESHOLD:
            concepts_above_threshold += 1
        concept_contribution = weight * adjusted
        weighted_score += concept_contribution
        concept_details.append({
            "concept_name": concept["concept_name"],
            "weight": weight,
            "best_sentence": best_sentence,
            "adjusted_similarity": round(adjusted, 4),
            "contradiction": round(contradiction_prob, 4),
        })

    base_score = (weighted_score / total_weight) * 100.0
    coverage_ratio = (concepts_above_threshold / len(rubric)) if len(rubric) > 0 else 0.0
    coverage_bonus = CONCEPT_COVERAGE_BONUS * coverage_ratio * 100.0
    return float(np.clip(base_score + coverage_bonus, 0.0, 100.0)), concept_details

def compute_structural_score(student_answer: str) -> float:
    words = student_answer.split()
    length = len(words)
    if length < 10: score = (length / 10.0) * 14.0
    elif length < 30: score = 14.0 + ((length - 10) / 20.0) * 30.0
    elif length < 60: score = 44.0 + ((length - 30) / 30.0) * 24.0
    elif length < 100: score = 68.0 + ((length - 60) / 40.0) * 8.0
    else: score = STRUCTURAL_CEILING
    return round(float(np.clip(score, 0.0, STRUCTURAL_CEILING)), 2)

def compute_hybrid_score(concept_score: float, structural_score: float, semantic_score: float, coverage_ratio: float, contradiction_ratio: float, off_topic_gate: float = 1.0):
    c = float(np.clip(concept_score, 0.0, 100.0))
    sem = float(np.clip(semantic_score, 0.0, 100.0))
    if coverage_ratio is not None and c < 36.0 and sem > 54.0 and coverage_ratio < 0.34:
        sem *= 0.50 + 0.50 * min(1.0, float(coverage_ratio) / 0.34)
    if off_topic_gate >= 0.96 and coverage_ratio is not None and coverage_ratio >= 0.36 and c < 52.0 and sem >= 48.0:
        c = min(76.0, c + min(15.0, max(0.0, 52.0 - c) * 0.35 + ((float(coverage_ratio) - 0.36) / 0.64) * 11.0))
    s = structural_score * (0.38 if concept_score < 22.0 else (0.64 if sem >= 52.0 else 0.58) if concept_score < 38.0 else 0.72 if concept_score < 52.0 else 1.0)
    hybrid = (WEIGHT_CONCEPT * c) + (WEIGHT_STRUCT * s) + (WEIGHT_SEMANTIC * sem)
    if coverage_ratio is not None:
        if coverage_ratio == 0.0: hybrid = min(hybrid, COVERAGE_CAP_NONE)
        elif coverage_ratio < 0.40: hybrid = min(hybrid, COVERAGE_CAP_40)
        elif coverage_ratio < 0.60: hybrid = min(hybrid, COVERAGE_CAP_60)
    hybrid -= CONTRADICTION_ADDITIVE_MAX * float(np.clip(contradiction_ratio, 0.0, 1.0))
    return round(float(np.clip(hybrid, 0.0, 100.0)), 2)

# -----------------------------------------------------------------------------
# RUN EVALUATION PIPELINE
# -----------------------------------------------------------------------------
print("=" * 70)
print("🎯 STARTING INTELLIPLACE EVALUATION")
print("=" * 70)

rubric = build_rubric_from_reference(REFERENCE_ANSWER)
print(f"📋 Generated Rubric Concepts ({len(rubric)} concepts):")
for idx, c in enumerate(rubric, 1):
    print(f"   {idx}. {c['concept_name']} (Weight: {int(c['weight']*100)}%)")

print("\\n🧠 Generating embeddings (handling multi-page answer)...")
student_doc_emb = get_hierarchical_doc_embedding(resolved_student_text)
ideal_doc_emb = bi_encoder.encode(REFERENCE_ANSWER, convert_to_numpy=True)

sentences = split_into_sentences(resolved_student_text)
print(f"📑 Split answer into {len(sentences)} sentences across all pages.")
sentence_embeddings = bi_encoder.encode(sentences, batch_size=32, show_progress_bar=False, convert_to_numpy=True)

semantic_score, off_topic_gate = compute_semantic_score(student_doc_emb, ideal_doc_emb)
concept_score, concept_details = compute_concept_score(sentences, sentence_embeddings, student_doc_emb, rubric)
gated_concept_score = concept_score * off_topic_gate
structural_score = compute_structural_score(resolved_student_text)

adj_sims = [float(d["adjusted_similarity"]) for d in concept_details]
thresh = COVERAGE_THRESHOLD
parts = [1.0 if v >= thresh else max(0.0, (v - (thresh - SOFT_COVERAGE_BAND)) / SOFT_COVERAGE_BAND) if v >= (thresh - SOFT_COVERAGE_BAND) else 0.0 for v in adj_sims]
coverage_ratio = (sum(parts) / float(len(parts))) if parts else 0.0

n_contradictions = sum(1 for d in concept_details if float(d.get("contradiction", 0.0)) >= CONTRADICTION_RATIO_THRESH)
contradiction_ratio = (n_contradictions / float(len(rubric))) if rubric else 0.0

final_score_100 = compute_hybrid_score(gated_concept_score, structural_score, semantic_score, coverage_ratio, contradiction_ratio, off_topic_gate)
final_score_10 = round(final_score_100 / 10.0, 2)

verdict = "🌟 Excellent Answer" if final_score_10 >= 8.5 else "👍 Good Understanding" if final_score_10 >= 7.0 else "⚠️ Partial Understanding" if final_score_10 >= 5.0 else "❌ Weak Answer"
verdict_color = "#28a745" if final_score_10 >= 8.5 else "#17a2b8" if final_score_10 >= 7.0 else "#ffc107" if final_score_10 >= 5.0 else "#dc3545"

table_rows = ""
strengths = []
weaknesses = []
for d in concept_details:
    score_pct = int(round(d["adjusted_similarity"] * 100))
    if score_pct >= 75:
        badge = '<span style="color:white;background:#28a745;padding:3px 8px;border-radius:4px;font-weight:bold;">FULL</span>'
        strengths.append(f"Strong coverage of <b>{d['concept_name']}</b>.")
    elif score_pct >= 40:
        badge = '<span style="color:black;background:#ffc107;padding:3px 8px;border-radius:4px;font-weight:bold;">PARTIAL</span>'
        weaknesses.append(f"<b>{d['concept_name']}</b> needs more technical depth.")
    else:
        badge = '<span style="color:white;background:#dc3545;padding:3px 8px;border-radius:4px;font-weight:bold;">MISSED</span>'
        weaknesses.append(f"Key concept <b>{d['concept_name']}</b> was missed.")

    con_flag = ' <br><span style="color:red;font-size:0.85em;">⚠️ Contradiction / Hallucination detected</span>' if d['contradiction'] >= CONTRADICTION_RATIO_THRESH else ''
    snippet = d['best_sentence'][:160] + "..." if len(d['best_sentence']) > 160 else d['best_sentence']
    table_rows += f\"\"\"<tr style="border-bottom:1px solid #ddd;">
        <td style="padding:10px;font-weight:600;">{d['concept_name']}</td>
        <td style="padding:10px;text-align:center;">{int(d['weight']*100)}%</td>
        <td style="padding:10px;text-align:center;font-weight:bold;">{score_pct}% {badge}</td>
        <td style="padding:10px;font-size:0.9em;color:#333;"><i>\"{snippet}\"</i>{con_flag}</td>
    </tr>\"\"\"

html_report = f\"\"\"
<div style="font-family:Arial,sans-serif;border:2px solid #e0e0e0;border-radius:12px;padding:25px;margin-top:20px;background:#fdfdfd;">
    <div style="display:flex;justify-content:space-between;align-items:center;border-bottom:2px solid #eee;padding-bottom:15px;">
        <div>
            <h2 style="margin:0;color:#2c3e50;">📊 IntelliPlace Evaluation Result</h2>
            <p style="margin:5px 0 0;color:#7f8c8d;"><b>Question:</b> {REFERENCE_QUESTION}</p>
        </div>
        <div style="text-align:right;">
            <div style="font-size:2.4em;font-weight:900;color:{verdict_color};">{final_score_10} <span style="font-size:0.5em;color:#888;">/ 10</span></div>
            <div style="font-size:1.1em;font-weight:bold;color:{verdict_color};">{verdict}</div>
        </div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0;">
        <div style="background:#eef7fe;padding:12px;border-radius:8px;text-align:center;"><div style="font-size:0.8em;color:#555;font-weight:bold;">CONCEPT SCORE</div><div style="font-size:1.4em;font-weight:bold;color:#007bff;">{concept_score:.1f}%</div></div>
        <div style="background:#f0fff4;padding:12px;border-radius:8px;text-align:center;"><div style="font-size:0.8em;color:#555;font-weight:bold;">SEMANTIC ALIGNMENT</div><div style="font-size:1.4em;font-weight:bold;color:#28a745;">{semantic_score:.1f}%</div></div>
        <div style="background:#fff9db;padding:12px;border-radius:8px;text-align:center;"><div style="font-size:0.8em;color:#555;font-weight:bold;">STRUCTURAL SCORE</div><div style="font-size:1.4em;font-weight:bold;color:#f39c12;">{structural_score:.1f}%</div></div>
        <div style="background:#f9f9f9;padding:12px;border-radius:8px;text-align:center;"><div style="font-size:0.8em;color:#555;font-weight:bold;">COVERAGE RATIO</div><div style="font-size:1.4em;font-weight:bold;color:#333;">{int(coverage_ratio*100)}%</div></div>
    </div>
    <h3 style="color:#2c3e50;">📑 Rubric Concept-by-Concept Breakdown</h3>
    <table style="width:100%;border-collapse:collapse;background:white;border-radius:8px;overflow:hidden;">
        <thead style="background:#f4f6f8;"><tr><th style="padding:10px;text-align:left;">Concept</th><th style="padding:10px;text-align:center;">Weight</th><th style="padding:10px;text-align:center;">Match</th><th style="padding:10px;text-align:left;">Best Evidence from Student Answer</th></tr></thead>
        <tbody>{table_rows}</tbody>
    </table>
</div>
\"\"\"
display(HTML(html_report))""")

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": "IntelliPlace_Answer_Grading_Engine.ipynb",
                "provenance": []
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open("IntelliPlace_Answer_Grading_Google_Colab.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print("✅ Created clean English notebook with auto-loading models!")

if __name__ == "__main__":
    create_notebook()
