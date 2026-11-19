"""
embedding.py
------------
SBERT bi-encoder and NLI model wrappers.

No changes from the previous version — these choices are correct:
- all-mpnet-base-v2: best general SBERT model for sentence similarity
- cross-encoder/nli-deberta-v3-small: better calibrated for short
  technical text than facebook/bart-large-mnli (which was the original
  wrong choice)

Label order for nli-deberta-v3-small:
  0 = entailment
  1 = neutral
  2 = contradiction
"""

import os

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OFFLINE_MODE = (
    os.environ.get("HF_HUB_OFFLINE") == "1"
    or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
)

# ---------------------------------------------------------------------------
# Bi-encoder
# ---------------------------------------------------------------------------
bi_encoder = SentenceTransformer("all-mpnet-base-v2", local_files_only=OFFLINE_MODE)

# ---------------------------------------------------------------------------
# NLI model
# ---------------------------------------------------------------------------
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
nli_tokenizer  = AutoTokenizer.from_pretrained(NLI_MODEL_NAME, local_files_only=OFFLINE_MODE)
nli_model      = AutoModelForSequenceClassification.from_pretrained(
    NLI_MODEL_NAME,
    local_files_only=OFFLINE_MODE,
    low_cpu_mem_usage=True,
)
nli_model.to(DEVICE)
nli_model.eval()


def get_embedding(text: str):
    """Return a numpy embedding vector for the given text."""
    return bi_encoder.encode(text)


def get_entailment_score(premise: str, hypothesis: str):
    """
    Returns (entailment_prob, contradiction_prob) as floats in [0, 1].

    Label order for nli-deberta-v3-small:
      index 0 = entailment
      index 1 = neutral
      index 2 = contradiction
    """
    inputs = nli_tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = nli_model(**inputs).logits

    probs = F.softmax(logits, dim=1)[0]

    entailment_prob    = probs[0].item()
    contradiction_prob = probs[2].item()

    return entailment_prob, contradiction_prob