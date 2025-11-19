import re

def normalize_text(text):

    #  Remove zero-width characters & BOM
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



'''
def normalize_text(text):
    # Remove zero-width chars and BOM
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)

    # Replace non-breaking space with normal space
    text = text.replace("\u00A0", " ")

    # Collapse only spaces + tabs, NOT newlines
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ newlines → 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
# ALTERNATIVE NORMALIZATION FUNCTION that collapses newlines
    
def normalize_text(text):
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    text = text.replace("\u00A0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
'''
