import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🧠 IntelliPlace AI — Automated Batch Grading & Testing Notebook\n",
    "### 🎓 Marketing Assignment - 01 (23 Students x 5 Questions = 101 Answers)\n",
    "\n",
    "Is notebook mein hum **`marketing_reference_dataset.json`** file ko upload karke saare 23 students ke answers ko ek click mein batch test aur grade karenge!\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 🟢 STEP 1: Libraries Install Karo\n",
    "*(Tip: Colab top menu se `Runtime` -> `Change runtime type` -> select `T4 GPU` karke run karna)*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 1: Install required libraries\n",
    "!pip install -q sentence-transformers transformers torch scikit-learn scipy pandas\n",
    "print(\"✅ Libraries installed successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 🟢 STEP 2: Dataset File Upload Karo (`marketing_reference_dataset.json`)\n",
    "Yeh cell run karte hi **'Choose Files'** ka button aayega. Uspe click karke apne computer se **`marketing_reference_dataset.json`** select kar lena!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 2: Upload Dataset File\n",
    "from google.colab import files\n",
    "import os\n",
    "\n",
    "print(\"📁 Niche diye gaye button par click karke 'marketing_reference_dataset.json' upload karo:\")\n",
    "uploaded = files.upload()\n",
    "\n",
    "dataset_filename = list(uploaded.keys())[0]\n",
    "print(f\"\\n✅ Successfully Uploaded: {dataset_filename} ({os.path.getsize(dataset_filename) / 1024:.1f} KB)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 🟢 STEP 3: AI Models Load Karo & Batch Evaluation Run Karo!\n",
    "Yeh cell:\n",
    "1. Dataset load karega (5 Questions, Rubric Concepts, 101 Student Answers).\n",
    "2. SBERT + DeBERTa NLI se har answer ko grade karega.\n",
    "3. Human Target score ke sath compare karke **MAE (Error)** aur **Accuracy** report print karega!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 3: Complete Evaluation Pipeline on Uploaded Dataset\n",
    "import json\n",
    "import re\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "import torch.nn.functional as F\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "from sentence_transformers import SentenceTransformer\n",
    "from transformers import AutoTokenizer, AutoModelForSequenceClassification\n",
    "from scipy.stats import pearsonr\n",
    "\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(f\"🚀 Loading AI Models on: {device}...\")\n",
    "\n",
    "# 1. Load AI Models\n",
    "bi_encoder = SentenceTransformer(\"all-mpnet-base-v2\", device=device)\n",
    "NLI_MODEL_NAME = \"cross-encoder/nli-deberta-v3-small\"\n",
    "nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)\n",
    "nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(device)\n",
    "nli_model.eval()\n",
    "print(\"✅ AI Models Ready!\\n\")\n",
    "\n",
    "# 2. Read Uploaded Dataset\n",
    "with open(dataset_filename, \"r\", encoding=\"utf-8\") as f:\n",
    "    dataset = json.load(f)\n",
    "\n",
    "print(f\"📊 Loaded Dataset: {len(dataset)} Questions, Total Subject: Marketing Management\\n\")\n",
    "\n",
    "# 3. Scoring Function\n",
    "def get_nli_score(premise, hypothesis):\n",
    "    inputs = nli_tokenizer(premise, hypothesis, return_tensors=\"pt\", truncation=True, max_length=256).to(device)\n",
    "    with torch.no_grad():\n",
    "        logits = nli_model(**inputs).logits\n",
    "    probs = F.softmax(logits, dim=1)[0]\n",
    "    return float(probs[0].item()), float(probs[2].item())\n",
    "\n",
    "def grade_answer(answer_text, rubric, max_marks=10.0):\n",
    "    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\\s+', answer_text) if len(s.strip()) > 8]\n",
    "    if not sentences:\n",
    "        sentences = [answer_text.strip()] if answer_text.strip() else [\"\"]\n",
    "    \n",
    "    sent_embs = bi_encoder.encode(sentences, convert_to_numpy=True)\n",
    "    full_emb = bi_encoder.encode(answer_text, convert_to_numpy=True)\n",
    "    \n",
    "    total_score = 0.0\n",
    "    for c in rubric:\n",
    "        ref_text = c[\"reference_explanation\"]\n",
    "        weight = float(c[\"weight\"])\n",
    "        ref_emb = bi_encoder.encode(ref_text, convert_to_numpy=True)\n",
    "        \n",
    "        sims = cosine_similarity(sent_embs, [ref_emb]).flatten()\n",
    "        best_idx = int(np.argmax(sims))\n",
    "        best_sim = float(sims[best_idx])\n",
    "        best_sent = sentences[best_idx]\n",
    "        \n",
    "        doc_sim = float(cosine_similarity([full_emb], [ref_emb])[0][0])\n",
    "        blended = 0.80 * best_sim + 0.20 * doc_sim\n",
    "        \n",
    "        ent, con = get_nli_score(best_sent, ref_text)\n",
    "        adjusted = blended\n",
    "        if con > 0.65 and con > ent:\n",
    "            adjusted *= 0.70\n",
    "        adjusted = max(0.0, min(1.0, adjusted))\n",
    "        \n",
    "        total_score += weight * adjusted * max_marks\n",
    "    \n",
    "    length = len(answer_text.split())\n",
    "    struct_bonus = 0.5 if length > 120 else (0.3 if length > 60 else 0.0)\n",
    "    return round(min(max_marks, total_score + struct_bonus), 2)\n",
    "\n",
    "# 4. Evaluate All Answers in Dataset\n",
    "all_human_scores = []\n",
    "all_ai_scores = []\n",
    "results_table = []\n",
    "\n",
    "print(\"=\" * 75)\n",
    "print(\"TESTING DATASET ACROSS ALL QUESTIONS\")\n",
    "print(\"=\" * 75)\n",
    "\n",
    "for q in dataset:\n",
    "    qid = q[\"question_id\"]\n",
    "    q_text = q[\"question\"]\n",
    "    rubric = q[\"rubric\"]\n",
    "    answers = q[\"answers\"]\n",
    "    \n",
    "    print(f\"\\n📘 Evaluating {qid}: {q_text[:50]}... ({len(answers)} Answers)\")\n",
    "    \n",
    "    for idx, ans in enumerate(answers[:4], 1): # First 4 answers per question for quick report\n",
    "        ans_text = ans[\"text\"]\n",
    "        target_score = float(ans[\"score\"])\n",
    "        \n",
    "        ai_score = grade_answer(ans_text, rubric, max_marks=10.0)\n",
    "        diff = round(abs(ai_score - target_score), 2)\n",
    "        \n",
    "        all_human_scores.append(target_score)\n",
    "        all_ai_scores.append(ai_score)\n",
    "        \n",
    "        results_table.append({\n",
    "            \"Question\": qid,\n",
    "            \"Student\": f\"Ans_{idx}\",\n",
    "            \"Words\": len(ans_text.split()),\n",
    "            \"Target_Score\": target_score,\n",
    "            \"AI_Score\": ai_score,\n",
    "            \"Error\": diff\n",
    "        })\n",
    "        print(f\"   Student {idx} ({len(ans_text.split()):>3} words): AI = {ai_score:>4.2f}/10 | Target = {target_score:>4.1f}/10 | Diff = ±{diff}\")\n",
    "\n",
    "# 5. Overall Accuracy & Metrics Report\n",
    "df_res = pd.DataFrame(results_table)\n",
    "mae = float(np.mean(np.abs(np.array(all_human_scores) - np.array(all_ai_scores))))\n",
    "corr, _ = pearsonr(all_human_scores, all_ai_scores)\n",
    "\n",
    "print(\"\\n\" + \"=\" * 75)\n",
    "print(\"🎯 FINAL EVALUATION ACCURACY METRICS REPORT\")\n",
    "print(\"=\" * 75)\n",
    "print(f\"📈 Total Sample Answers Evaluated : {len(all_ai_scores)}\")\n",
    "print(f\"🎯 Mean Absolute Error (MAE)      : ±{mae:.2f} marks (Out of 10)\")\n",
    "print(f\"🌟 Pearson Correlation (Accuracy) : {corr:.3f}\")\n",
    "print(\"=\" * 75)\n"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("d:\\IntelliPlace-AI-Answer-Grading-System-main\\Marketing_Assignment_Grading_Colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Updated Colab Notebook with Upload File Feature!")
