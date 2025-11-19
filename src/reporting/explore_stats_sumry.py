import json
import random
from collections import Counter
import os
import numpy as np
from detectors.html_detect import has_html
from detectors.language_detect import detect_lang
from detectors.code_ASCII_detect import code_fraction
from detectors.code_strong_detect import code_fraction_strong


def quick_stats(filepath, sample_size=20000):
    limit=sample_size
    key_counts = Counter()
    length_distribution = []
    empty_count = 0
    short_count = 0

    # file size
    file_size_bytes = os.path.getsize(filepath)
    file_size_mb = file_size_bytes / (1024 * 1024)
    file_size_gb = file_size_mb / 1024

    #  Count total lines
    with open(filepath, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # test randomly drawn samples
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit: 
                break
            
            try:
                row = json.loads(line)
            except:
                continue #skip malformed rows
            
            key_counts.update(row.keys())
            text = row.get("text", "")
            
            len_txt=len(text)
            length_distribution.append(len_txt)
            if len_txt ==0:
                empty_count += 1
            if len_txt<100:  #short text threshold
                short_count += 1


    print("FILE OVERVIEW")
    print(f"File size: {file_size_mb:.2f} MB ({file_size_gb:.2f} GB)")
    print(f"Total lines in file: {total_lines:,}")
    print(f"Sampled (limit): {min(limit, total_lines):,}")
    print("------------")
    print("\nKEYS")
    print("Most common keys:", key_counts.most_common(10))
    print("------------")
    print("\nTEXT LENGTH STATS")

    arr = np.array(length_distribution)
    print(f"Avg length: {arr.mean():.2f}")
    print(f"Median length: {np.median(arr):.2f}")
    print(f"10th percentile: {np.percentile(arr, 10):.2f}")
    print(f"90th percentile: {np.percentile(arr, 90):.2f}")
    if np.median(arr) > 300 and arr.mean() > 600:
        print("Most entries are multi-paragraph documents (median text size>300).")
        print("Good for LLM pretraining (rich context, natural text).")
    else:
        print("Overall text seems short on average. Many entries may be fragments, comments, or low-quality text.")

    print(f"Max length: {arr.max()}")
    if arr.max() > 20000:
        print("WARNING: Very long entries detected (>> 20,000 chars). These are almost certainly web dumps such as, " )
        print("raw HTML pages,  code dumps / stack traces, JSON logs or config files, full chat transcripts, base64 or encoded junk." )
        print(" These must be  filtered or trim these before LLM pretraining")
    print(f"Min length: {arr.min()}")
    print("------------")
    print("\nNOISE INDICATORS")
    print(f"Empty texts: {empty_count} ({empty_count/len(arr):.2%})")
    short_txt_pct=short_count/len(arr)
    print(f"Short < 100 chars: {short_count} ({short_txt_pct:.2%})")
    if short_txt_pct < 10:
        print("Only a small fraction of trivial/low-value text.  Dataset likely contains substantial natural text, not noise")
    else:
        print("Many very short entries, may contain garbage or low-quality data.")

    print("------------")
    print("\nDone.")


def show_longest_docs(path, n=5):
    docs = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text","")
            docs.append((len(text), text))
    docs.sort(reverse=True)
    for i, (l, t) in enumerate(docs[:n], 1):
        print(f"\n=== Longest Doc #{i} ({l} chars) ===\n")
        print(t[:2000])
        print("\n------------------------------------\n")


def summarize_dataset(path, sample_size=10000):
    summary = Counter()

    with open(path, "r", encoding="utf-8") as f:
        lines = random.sample(list(f), sample_size)

    for line in lines:
        try:
            row = json.loads(line)
        except:
            summary["MALFORMED"] += 1
            continue

        text = row.get("text", "")
        if not text.strip():
            summary["EMPTY"] += 1
            continue

        # Language detection
        lang = detect_lang(text)
        if lang != "EN":
            summary["NON_ENGLISH"] += 1

        # HTML detection
        if has_html(text):
            summary["HTML"] += 1

        # Code-heavy detection
        if code_fraction(text) > 0.40:
            summary["CODE_HEAVY"] += 1

        # Good English
        if (
            lang == "EN"
            and not has_html(text)
            and code_fraction(text) <= 0.40
            and len(text) >= 200
        ):
            summary["GOOD_ENGLISH"] += 1

    # convert to percentages
    summary_pct = {k: (v / sample_size) * 100 for k, v in summary.items()}

    return summary, summary_pct



def classify_doc(text):
    """Return a single category for each document."""
    
    # HTML has highest priority
    if has_html(text):
        return "HTML"
    
    # Code-heavy comes next
    #code_fraction_strong or code_fraction
    if code_fraction_strong(text) > 0.40:
        return "CODE_HEAVY"
    
    # OTHER Languages
    lang = detect_lang(text)
    if lang != "EN":
        return "NON_ENGLISH_LANG"
    
    # Good English (no HTML, no code, significant length for training, EN) 
    if len(text) >= 200:
        return "GOOD_ENGLISH"
    
    # 5 — fallback
    return "SHORT_ENGLISH"

def summarize_dataset_exclusive(path, sample_size=10000):
    summary = Counter()

    with open(path, "r", encoding="utf-8") as f:
        lines = random.sample(list(f), sample_size)

    for line in lines:
        try:
            row = json.loads(line)
        except:
            summary["MALFORMED"] += 1
            continue

        text = row.get("text", "").strip()
        if not text:
            summary["EMPTY"] += 1
            continue

        category = classify_doc(text)
        summary[category] += 1

    # Convert to percentages
    summary_pct = {k: v / sample_size * 100 for k, v in summary.items()}

    return summary, summary_pct
