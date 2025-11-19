import json
import re
import html


# Detects real HTML tags
HTML_TAG_PATTERN = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(\s+[^>]*)?>")

# Detects escaped & entities
HTML_ENTITY_PATTERN = re.compile(r"&[A-Za-z]+;|&#\d+;")

# Detect <script> or <style> blocks
SCRIPT_STYLE_PATTERN = re.compile(r"<(script|style)[\s>]", re.IGNORECASE)

def has_html(text):
    """Return True if text contains any HTML tags, escaped HTML, or script/style blocks."""
    if HTML_TAG_PATTERN.search(text):
        return True
    if HTML_ENTITY_PATTERN.search(text):
        return True
    if SCRIPT_STYLE_PATTERN.search(text):
        return True
    return False


def show_html_examples(path, n=5):
    shown = 0
    with open(path) as f:
        for line in f:
            if shown >= n:
                break
            row = json.loads(line)
            text = row.get("text","")
            if has_html(text):
                print("\n=== HTML DOC ===\n")
                print(text[:1500])
                print("\n----------------\n")
                shown += 1


def strip_html(text):
    # Remove script/style
    text = re.sub(
        r"<script.*?>.*?</script>|<style.*?>.*?</style>",
        " ", text, flags=re.DOTALL | re.IGNORECASE
    )

    # Remove comments
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)

    # Remove tags
    text = re.sub(r"</?[A-Za-z][A-Za-z0-9]*(\s+[^>]*)?>", " ", text)

    # Decode entities
    text = html.unescape(text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()
