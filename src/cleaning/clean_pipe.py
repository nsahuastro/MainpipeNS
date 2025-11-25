import json
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.utils.io_utils import stream_jsonl, write_jsonl
from src.detectors.html_detect import has_html, strip_html
from src.detectors.language_detect import detect_lang
from src.detectors.code_ASCII_detect import code_fraction
from src.detectors.code_strong_detect import code_fraction_strong

from src.cleaning.txt_norm_pipe import normalize_text

def clean_dataset(input_path, output_path, verbose=True):

    if verbose:
        print("\n=== RUNNING CLEANING PIPELINE ===")

    counters = Counter()

    with open(output_path, "w", encoding="utf-8") as fout:
        for row in tqdm(stream_jsonl(input_path)):
            try:
                text = row.get("text", "").strip()
            except:
                counters["MALFORMED"] += 1
                continue

            # 1. Empty
            if not text:
                counters["EMPTY"] += 1
                continue

            # 2. Strip HTML
            if has_html(text):
                counters["HTML_STRIPPED"] += 1
                text = strip_html(text)

            # 3. Language detection
            lang = detect_lang(text)
            if lang != "EN":
                counters["NON_ENGLISH"] += 1
                continue

            # 4. Code-heavy filtering
            #if code_fraction(text) > 0.40:
            #    counters["CODE_HEAVY"] += 1
            #    continue
            if code_fraction_strong(text) > 0.40:
                counters["CODE_HEAVY"] += 1
                continue

            # 5. Length rules
            L = len(text)
            if L < 200:
                counters["TOO_SHORT"] += 1
                continue
            if L > 50000:
                counters["TOO_LONG"] += 1
                continue

            # 6. Normalize
            text = normalize_text(text)

            # Save cleaned doc
            fout.write(json.dumps({"text": text}) + "\n")
            counters["KEPT"] += 1

    total = sum(counters.values())
    if verbose:
        print("\n=== CLEANING FINISHED ===")
        for k, v in counters.items():
            print(f"{k}: {v:,} ({v*100/total:6.2f}%) ")

    return counters


def print_cleaning_summary(counters):
    print("\n========== CLEANING SUMMARY ==========")
    total = sum(counters.values())

    # Explicitly print removals first
    removal_keys = [
        "EMPTY", "HTML_STRIPPED", "NON_ENGLISH", "CODE_HEAVY",
        "TOO_SHORT", "TOO_LONG", "MALFORMED"
    ]
    keep_key = "KEPT"

    for k in removal_keys:
        if k in counters:
            print(f"{k:15}: {counters[k]:6}  ({counters[k]/total*100:6.2f}%)")

    if keep_key in counters:
        print(f"{keep_key:15}: {counters[keep_key]:6}  ({counters[keep_key]/total*100:6.2f}%)")

    print("======================================\n")

    
