import re

def normalize_text(text):

    #  Remove zero-width characters and BOM
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)

    #  Replace non-breaking space
    text = text.replace("\u00A0", " ")

    #  Remove weird control characters (except \n \t)
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", " ", text)

    #  Collapse multiple spaces/tabs (but **not** newlines)
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ newlines into 2 (paragraph normalization)
    text = re.sub(r"\n{3,}", "\n\n", text)

    #  Strip leading / trailing whitespace
    return text.strip()



