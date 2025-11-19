import json
import random
from lingua import Language, LanguageDetectorBuilder
from collections import Counter


ALL_LANGUAGES = Language.all()
lang_detector = LanguageDetectorBuilder.from_languages(*ALL_LANGUAGES).build()

def detect_lang(text):
    try:
        lang = lang_detector.detect_language_of(text)
        return lang.iso_code_639_1.name if lang else "UNKNOWN"
    except:
        return "ERROR"

def sample_language_distribution(path, sample_size=5000):
    langs = Counter()
    lines = []

    with open(path, "r", encoding="utf-8") as f:
        lines = random.sample(list(f), sample_size)

    for line in lines:
        row = json.loads(line)
        text = row.get("text", "")
        if not text.strip():
            continue

        lang = detect_lang(text)
        langs[lang] += 1

    return langs