import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import re
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Loading models on', device)
bi_encoder = SentenceTransformer('all-mpnet-base-v2', device=device)
NLI_MODEL_NAME = 'cross-encoder/nli-deberta-v3-small'
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME, low_cpu_mem_usage=True).to(device)
nli_model.eval()

# Load assignment
OFFICIAL_ASSIGNMENT = [
    {
        'q_num': 1,
        'question': 'Question 1: Define Marketing. Explain the nature of marketing along with examples.',
        'max_marks': 10.0,
        'rubric': [
            {'concept_name': 'Definition of Marketing', 'weight': 0.20, 'reference_explanation': 'Marketing is the social and managerial process of identifying customer needs, creating products or services, communicating their value, and delivering satisfaction to customers while achieving organizational goals.'},
            {'concept_name': 'Nature of Marketing', 'weight': 0.50, 'reference_explanation': 'Customer-Oriented, Continuous Process, Value-Centered, Integrated Activity, Exchange-Based, Goal-Oriented, Dynamic.'},
            {'concept_name': 'Relevant Real-world Examples', 'weight': 0.30, 'reference_explanation': 'Practical examples like streaming platforms, food delivery apps, smartphone features, laptop company.'}
        ]
    },
    {
        'q_num': 2,
        'question': 'Question 2: What all are the scope of marketing & importance of marketing?',
        'max_marks': 10.0,
        'rubric': [
            {'concept_name': 'Scope of Marketing (5-6 functional areas with examples)', 'weight': 0.50, 'reference_explanation': 'Market Research, Product Planning & Development, Branding, Pricing, Promotion, Distribution/Place with examples.'},
            {'concept_name': 'Importance of Marketing (5 benefits with examples)', 'weight': 0.50, 'reference_explanation': 'Creates Awareness, Increases Sales & Profit, Ensures Customer Satisfaction, Builds Brand Image & Loyalty, Helps Survive Competition with examples.'}
        ]
    },
    {
        'q_num': 3,
        'question': 'Question 3: Differentiate between needs & wants. Explain how marketing influences both.',
        'max_marks': 10.0,
        'rubric': [
            {'concept_name': 'Difference between Needs and Wants (Table)', 'weight': 0.50, 'reference_explanation': 'Needs are basic human requirements essential for survival (food, water, shelter, clothing), whereas wants are desires for specific goods or services shaped by culture, personality, and lifestyle (pizza, branded clothes, iPhone).'},
            {'concept_name': 'How Marketing Influences Both (4 mechanisms)', 'weight': 0.50, 'reference_explanation': 'Through Advertising, Through Branding, Through Brand Image & Lifestyle Positioning, Through Attractive Packaging.'}
        ]
    },
    {
        'q_num': 4,
        'question': 'Question 4: Explain Marketing Environment & Define Internal and External Marketing Environment.',
        'max_marks': 10.0,
        'rubric': [
            {'concept_name': 'Concept & Characteristics of Marketing Environment', 'weight': 0.30, 'reference_explanation': 'Sum total of internal and external forces influencing marketing decisions. Characteristics: Dynamic, Complex, Uncontrollable.'},
            {'concept_name': 'Internal Marketing Environment', 'weight': 0.35, 'reference_explanation': 'Forces within organization directly controllable by management: Top Management policies, Employees, Organizational Structure, Company Culture, Financial/R&D Resources.'},
            {'concept_name': 'External Marketing Environment (Micro vs Macro)', 'weight': 0.35, 'reference_explanation': 'Forces outside organization with little/no direct control. Divided into Micro Environment (customers, suppliers, competitors, intermediaries, publics) and Macro Environment (demographic, economic, socio-cultural, technological, political/legal).'}
        ]
    },
    {
        'q_num': 5,
        'question': 'Question 5: Explain Micro & Macro external environmental factors.',
        'max_marks': 10.0,
        'rubric': [
            {'concept_name': 'Micro External Environmental Factors', 'weight': 0.50, 'reference_explanation': 'Task-environment forces close to company: Customers, Suppliers, Competitors, Marketing Intermediaries, Publics.'},
            {'concept_name': 'Macro External Environmental Factors', 'weight': 0.50, 'reference_explanation': 'Broader uncontrollable societal forces: Demographic Factors, Economic Factors, Social & Cultural Factors, Technological Factors, Political & Legal Factors.'}
        ]
    }
]

with open('student_answer_40plus.txt', encoding='utf-8') as f:
    student_text = f.read()

split_patterns = [
    (1, r'(?:Question[-:\s]*1)'),
    (2, r'(?:Question[-:\s]*2)'),
    (3, r'(?:Question[-:\s]*3)'),
    (4, r'(?:Question[-:\s]*4)'),
    (5, r'(?:Question[-:\s]*5)')
]
indices = []
for qn, pat in split_patterns:
    m = re.search(pat, student_text, re.IGNORECASE)
    if m:
        indices.append((qn, m.start()))
indices.sort(key=lambda x: x[1])
student_answers = {}
for i in range(len(indices)):
    qn, s = indices[i]
    e = indices[i+1][1] if i+1 < len(indices) else len(student_text)
    student_answers[qn] = student_text[s:e].strip()

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

def split_into_sentences(text):
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    clean = [s.strip() for s in raw if len(s.strip()) > 8 and len(s.strip().split()) >= 2]
    return clean if clean else ([text.strip()] if text.strip() else [])

def get_hierarchical_doc_embedding(text):
    words = text.split()
    if len(words) <= 180:
        return bi_encoder.encode(text, convert_to_numpy=True)
    chunks = []
    start = 0
    while start < len(words):
        chunk = ' '.join(words[start:start + 180])
        chunks.append(chunk)
        start += (180 - 40)
    chunk_embs = bi_encoder.encode(chunks, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    doc_emb = np.mean(chunk_embs, axis=0)
    norm = np.linalg.norm(doc_emb)
    return (doc_emb / norm) if norm > 0 else doc_emb

def _hinge_remap(raw_sim):
    if raw_sim < HINGE_LOW: return 0.0
    elif raw_sim < HINGE_MID: return ((raw_sim - HINGE_LOW) / (HINGE_MID - HINGE_LOW)) * 0.40
    elif raw_sim < HINGE_HIGH: return 0.40 + ((raw_sim - HINGE_MID) / (HINGE_HIGH - HINGE_MID)) * 0.40
    else: return 0.80 + ((raw_sim - HINGE_HIGH) / (1.0 - HINGE_HIGH)) * 0.20

def get_entailment_score(premise, hypothesis):
    inputs = nli_tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, max_length=256).to(device)
    with torch.no_grad(): logits = nli_model(**inputs).logits
    probs = F.softmax(logits, dim=1)[0]
    return float(probs[0].item()), float(probs[2].item())

def compute_semantic_score(student_doc_emb, ideal_doc_emb):
    raw_similarity = float(cosine_similarity([student_doc_emb], [ideal_doc_emb])[0][0])
    raw_similarity = max(0.0, raw_similarity)
    corrected = float(np.clip((raw_similarity - SIMILARITY_BASELINE) / (1.0 - SIMILARITY_BASELINE), 0.0, 1.0))
    score = round(corrected * 100.0, 2)
    gate_signal = 0.58 * raw_similarity + 0.42 * corrected
    off_topic_gate = GATE_MAX if gate_signal >= OFF_TOPIC_THRESHOLD else (GATE_MIN + (gate_signal / OFF_TOPIC_THRESHOLD) * (GATE_MAX - GATE_MIN))
    return score, float(np.clip(off_topic_gate, GATE_MIN, GATE_MAX))

def compute_concept_score(sentences, sentence_embeddings, full_answer_emb, rubric):
    total_weight = sum(c.get('weight', 1.0) for c in rubric) or 1.0
    weighted_score = 0.0
    concept_details = []
    concepts_above_threshold = 0
    used_primary_indices = set()
    n_sent = len(sentences)
    n_con = len(rubric)
    slack = max(0.0, float(n_con + 2 - n_sent))
    reuse_penalty_mult = PRIMARY_SENTENCE_REUSE_PENALTY + (1.0 - PRIMARY_SENTENCE_REUSE_PENALTY) * min(1.0, slack / 3.0)

    for concept in rubric:
        ref_text = concept['reference_explanation']
        weight = float(concept.get('weight', 1.0))
        ref_embedding = bi_encoder.encode(ref_text, convert_to_numpy=True)
        sims_ref = cosine_similarity(sentence_embeddings, [ref_embedding]).flatten()
        raw_similarities = np.clip(sims_ref, 0.0, 1.0)
        pick_scores = raw_similarities.astype(float).copy()
        for idx in used_primary_indices: pick_scores[idx] *= reuse_penalty_mult
        best_idx = int(np.argmax(pick_scores))
        max_raw_sim = float(raw_similarities[best_idx])
        best_sentence = sentences[best_idx]
        used_primary_indices.add(best_idx)

        k = min(TOP_K_SENTENCES, len(sentences))
        top_indices = np.argsort(raw_similarities)[-k:][::-1]
        weights_used = TOP_K_WEIGHTS[:k]
        weights_used = [w / sum(weights_used) for w in weights_used]
        top_k_quality = sum(weights_used[rank] * float(raw_similarities[idx]) for rank, idx in enumerate(top_indices))

        doc_sim = float(np.clip(cosine_similarity([full_answer_emb], [ref_embedding])[0][0], 0.0, 1.0))
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
        if adjusted >= COVERAGE_THRESHOLD: concepts_above_threshold += 1
        concept_contribution = weight * adjusted
        weighted_score += concept_contribution
        concept_details.append({
            'concept_name': concept['concept_name'],
            'weight': weight,
            'adjusted_similarity': round(adjusted, 4),
            'contradiction': round(contradiction_prob, 4),
        })

    base_score = (weighted_score / total_weight) * 100.0
    cov_ratio = (concepts_above_threshold / len(rubric)) if len(rubric) > 0 else 0.0
    return float(np.clip(base_score + (CONCEPT_COVERAGE_BONUS * cov_ratio * 100.0), 0.0, 100.0)), concept_details

def compute_structural_score(student_answer):
    length = len(student_answer.split())
    if length < 10: s = (length / 10.0) * 14.0
    elif length < 30: s = 14.0 + ((length - 10) / 20.0) * 30.0
    elif length < 60: s = 44.0 + ((length - 30) / 30.0) * 24.0
    elif length < 100: s = 68.0 + ((length - 60) / 40.0) * 8.0
    else: s = STRUCTURAL_CEILING
    return round(float(np.clip(s, 0.0, STRUCTURAL_CEILING)), 2)

def compute_hybrid_score(c_score, s_score, sem_score, cov_ratio, contra_ratio, gate=1.0):
    c = float(np.clip(c_score, 0.0, 100.0))
    sem = float(np.clip(sem_score, 0.0, 100.0))
    if cov_ratio is not None and c < 36.0 and sem > 54.0 and cov_ratio < 0.34:
        sem *= 0.50 + 0.50 * min(1.0, float(cov_ratio) / 0.34)
    if gate >= 0.96 and cov_ratio is not None and cov_ratio >= 0.36 and c < 52.0 and sem >= 48.0:
        c = min(76.0, c + min(15.0, max(0.0, 52.0 - c) * 0.35 + ((float(cov_ratio) - 0.36) / 0.64) * 11.0))
    s = s_score * (0.38 if c_score < 22.0 else (0.64 if sem >= 52.0 else 0.58) if c_score < 38.0 else 0.72 if c_score < 52.0 else 1.0)
    hybrid = (WEIGHT_CONCEPT * c) + (WEIGHT_STRUCT * s) + (WEIGHT_SEMANTIC * sem)
    if cov_ratio is not None:
        if cov_ratio == 0.0: hybrid = min(hybrid, COVERAGE_CAP_NONE)
        elif cov_ratio < 0.40: hybrid = min(hybrid, COVERAGE_CAP_40)
        elif cov_ratio < 0.60: hybrid = min(hybrid, COVERAGE_CAP_60)
    hybrid -= CONTRADICTION_ADDITIVE_MAX * float(np.clip(contra_ratio, 0.0, 1.0))
    return round(float(np.clip(hybrid, 0.0, 100.0)), 2)

grand_total = 0.0
print('\n================ EVALUATION RESULTS ================')
for item in OFFICIAL_ASSIGNMENT:
    qn = item['q_num']
    rubric = item['rubric']
    max_m = item['max_marks']
    stu_ans = student_answers[qn]
    
    combined_ref = ' '.join([c['reference_explanation'] for c in rubric])
    student_doc_emb = get_hierarchical_doc_embedding(stu_ans)
    ideal_doc_emb = bi_encoder.encode(combined_ref, convert_to_numpy=True)
    sentences = split_into_sentences(stu_ans)
    sentence_embeddings = bi_encoder.encode(sentences, batch_size=32, show_progress_bar=False, convert_to_numpy=True)

    sem_score, gate = compute_semantic_score(student_doc_emb, ideal_doc_emb)
    c_score, details = compute_concept_score(sentences, sentence_embeddings, student_doc_emb, rubric)
    gated_c_score = c_score * gate
    struct_score = compute_structural_score(stu_ans)

    adjs = [float(d['adjusted_similarity']) for d in details]
    parts = [1.0 if v >= COVERAGE_THRESHOLD else max(0.0, (v - (COVERAGE_THRESHOLD - SOFT_COVERAGE_BAND)) / SOFT_COVERAGE_BAND) if v >= (COVERAGE_THRESHOLD - SOFT_COVERAGE_BAND) else 0.0 for v in adjs]
    cov_ratio = (sum(parts) / float(len(parts))) if parts else 0.0
    n_contra = sum(1 for d in details if float(d.get('contradiction', 0.0)) >= CONTRADICTION_RATIO_THRESH)
    contra_ratio = (n_contra / float(len(rubric))) if rubric else 0.0

    final_pct = compute_hybrid_score(gated_c_score, struct_score, sem_score, cov_ratio, contra_ratio, gate)
    marks_awarded = round((final_pct / 100.0) * max_m, 2)
    grand_total += marks_awarded
    print(f'Q{qn}: {marks_awarded} / {max_m} ({final_pct:.1f}%) | Concept: {c_score:.1f}% | Semantic: {sem_score:.1f}%')

print(f'\n🎯 GRAND TOTAL: {grand_total:.2f} / 50.0 ({grand_total/50.0*100:.1f}%)')
