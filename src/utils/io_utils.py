import json
import os
import random

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def read_jsonl(filepath, n=5):
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            print(json.loads(line))

def sample_jsonl(filepath, n=20):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    samples = random.sample(lines, n)
    
    print("---- Sampled Rows ----")
    for line in samples:
        print(json.loads(line))
        print("-------------")


def stream_jsonl(path):
    """Yield each JSON object from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except:
                continue

def write_jsonl(path, row):
    """Append a single JSON object as a JSONL line."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

def sample_docs(path, n=3):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        lines = random.sample(list(f), n)
    return [json.loads(l)["text"] for l in lines]
