import json
import re
import time
from collections import Counter
from detoxify import Detoxify
#from langdetect import detect
from src.detectors.language_detect import detect_lang
from src.utils.io_utils import sample_docs
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import numpy as np


# SIMPLE PII DETECTION 
def detect_pii(text):
    pii_found = []

    # emails
    if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
        pii_found.append("email")

    # phone numbers (very simple rule)
    if re.search(r"\b(\+?\d[\d\-\s]{7,}\d)\b", text):
        pii_found.append("phone")

    # credit card-like
    if re.search(r"\b(?:\d[ -]*?){13,16}\b", text):
        pii_found.append("credit_card")

    return pii_found


#  TOXICITY 
tox_model = Detoxify('original')

def toxicity_score(text):
    try:
        out = tox_model.predict(text)
        return out["toxicity"]
    except:
        return None


#  PERPLEXITY PROXY 
device = "cuda" if torch.cuda.is_available() else "cpu"

gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_model.eval()

@torch.no_grad()
def gpt2_perplexity(text):
    try:
        ids = gpt2_tok(text, return_tensors="pt", truncation=True).input_ids.to(device)
        output = gpt2_model(input_ids=ids, labels=ids)
        loss = output.loss.item()
        return float(torch.exp(torch.tensor(loss)))
    except:
        return None


def to_python(obj):
    """Convert numpy types to Python native types recursively."""
    if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    return obj


#  MAIN QUALITY REPORT 
def quality_report(path, sample_size=2000, save_path="reports/quality_report.json"):
    """
    Evaluate PII, toxicity, perplexity, and language distribution on CLEANED data.
    """

    docs = sample_docs(path, n=sample_size)

    pii_counter = Counter()
    tox_scores = []
    ppl_scores = []
    lang_counter = Counter()

    t0 = time.time()

    for text in docs:
        # PII
        pii = detect_pii(text)
        for p in pii:
            pii_counter[p] += 1

        # Toxicity
        tox = toxicity_score(text)
        if tox is not None:
            tox_scores.append(tox)

        # Perplexity
        ppl = gpt2_perplexity(text)
        if ppl is not None:
            ppl_scores.append(ppl)

        # Language
        lang = detect_lang(text)
        lang_counter[lang] += 1

    elapsed = time.time() - t0

    report = {
        "samples_analyzed": sample_size,
        "pii_hits": dict(pii_counter),
        "toxicity": {
            "avg": sum(tox_scores)/len(tox_scores) if tox_scores else None,
            "max": max(tox_scores) if tox_scores else None
        },
        "perplexity": {
            "avg": sum(ppl_scores)/len(ppl_scores) if ppl_scores else None,
            "median": float(torch.tensor(ppl_scores).median()) if ppl_scores else None
        },
        "language_distribution": dict(lang_counter),
        "analysis_time_sec": round(elapsed, 2)
    }

    clean_report = to_python(report)

    # Save JSON
    with open(save_path, "w") as f:
        json.dump(clean_report, f, indent=2)

    print(f"[quality_report] Saved to {save_path}")
    return clean_report
    