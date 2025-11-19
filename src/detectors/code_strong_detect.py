import re

# CODE-SPECIFIC SIGNALS
# Many programming languages use ; to end statements
SEMICOLON_LINE = re.compile(r";\s*$")

# Curly braces are extremely common in JS, C, Java, Rust, Go
BRACE_LINE = re.compile(r"[{}]")

# Common assignment operators
ASSIGNMENT = re.compile(r"\s(\w+)\s*=\s*[^=]")

# Code-like keywords across many languages
CODE_KEYWORDS = re.compile(
    r"\b("
    r"function|var|let|const|class|return|import|export|enum|extends|implements|"
    r"namespace|public|private|protected|static|void|try|catch|finally|throw|new|"
    r"#include|template|typename|struct|typedef|using|switch|case|break|continue|"
    r"async|await|yield|lambda|fn|match|crate|pub|impl|trait"
    r")\b"
)

# Python-specific patterns
PYTHON_DEF = re.compile(r"^\s*def\s+\w+\s*\(")
PYTHON_IMPORT = re.compile(r"^\s*(from\s+\w+|import\s+\w+)")

# Fenced code blocks (Markdown, StackOverflow, blogs)
FENCED_CODE = re.compile(r"```|<code>|</code>|<pre>|</pre>")

# Lines dominated by symbols
SYMBOL_HEAVY = re.compile(r"^[^A-Za-z0-9]{4,}$")


# MAIN LINE-LEVEL DETECTOR

def is_code_line_strong(line: str) -> bool:
    line = line.strip()

    # Empty line → not code
    if not line:
        return False

    # Fenced code markers
    if FENCED_CODE.search(line):
        return True

    # Starts with indentation typical for code (NOT English paragraphs)
    if line.startswith(("    ", "\t")):
        if re.search(r"[A-Za-z]\w*\s*\(", line):
            return True

    # Pure symbol lines
    if SYMBOL_HEAVY.match(line):
        return True

    # ; at end of line → JS/Java/C/Rust/Go
    if SEMICOLON_LINE.search(line):
        return True

    # Curly braces
    if BRACE_LINE.search(line):
        return True

    # Assignments like: x = 10
    if ASSIGNMENT.search(line):
        return True

    # Common code keywords
    if CODE_KEYWORDS.search(line):
        return True

    # Python patterns
    if PYTHON_DEF.match(line) or PYTHON_IMPORT.match(line):
        return True

    return False


# DOCUMENT-LEVEL CODE FRACTION

def code_fraction_strong(text: str) -> float:
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0

    code_lines = sum(1 for l in lines if is_code_line_strong(l))
    return code_lines / len(lines)



# YES/NO CODE CLASSIFICATION
def is_code_document_strong(text: str, threshold: float = 0.40) -> bool:
    return code_fraction_strong(text) > threshold
