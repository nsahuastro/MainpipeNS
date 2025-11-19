import json
import random
import re

# Strong signal Python code
PATTERN_PYTHON = re.compile(
    r"(^|\n)\s*def\s+\w+\s*\(|(^|\n)\s*return\b"
)

# Strong signal JavaScript/TypeScript
PATTERN_JS = re.compile(
    r"(^|\n)\s*function\s+\w+\s*\("   # function foo(
)

# Strong signal C/C++
PATTERN_CPP = re.compile(
    r"#include\s+<\w+\.h>"  # #include <stdio.h>
)

# Strong signal Rust
PATTERN_RUST = re.compile(
    r"(^|\n)\s*fn\s+\w+\s*\("  # fn main(
)

# Strong signal Java/C#
PATTERN_JAVA = re.compile(
    r"(^|\n)\s*(public|private|protected)\s+\w+\s+\w+\s*\("  # public void main(
)

def is_code_line(line):
    return (
        PATTERN_PYTHON.search(line) or
        PATTERN_JS.search(line) or
        PATTERN_CPP.search(line) or
        PATTERN_RUST.search(line) or
        PATTERN_JAVA.search(line)
    )


def sample_code_fraction(path, sample_size=5000):

    with open(path, "r") as f:
        lines = random.sample(list(f), sample_size)

    code_docs = 0
    for line in lines:
        row = json.loads(line)
        text = row.get("text", "")
        if code_fraction(text) > 0.40:
            code_docs += 1

    print(f"Code-heavy (>40% code) docs: {code_docs}/{sample_size} = {code_docs/sample_size:.2%}")


def code_fraction(text):
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    
    code_lines = sum(1 for l in lines if is_code_line(l))
    return code_lines / len(lines)


def detect_non_ascii(path, sample_size=20000):
    limit=sample_size
    non_english = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit: 
                break
            text = json.loads(line).get("text", "")
            
            # heuristic for non-English: presence of many non-Latin chars
            if re.search(r"[^\x00-\x7F]+", text):
                non_english += 1

    print(f"Likely non-English: {non_english}/{limit} = {non_english/limit:.2%}")