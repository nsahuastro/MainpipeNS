import re
import hashlib
from simhash import Simhash

def hash_text(text):
    """Return a stable SHA-256 hash of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenize_for_simhash(text):
    """
    Normalize text for SimHash:
    - lowercase
    - replace punctuation with spaces
    - collapse whitespace
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)  # safer than your previous version
    tokens = text.split()
    return tokens


def chunk_tokens(tokens, n=8):
    """Group tokens into chunks to improve simhash sensitivity."""
    return [" ".join(tokens[i:i+n]) for i in range(0, len(tokens), n)]


def simhash_text(text):
    """
    Compute SimHash of normalized token chunks (recommended).
    """
    tokens = tokenize_for_simhash(text)
    chunks = chunk_tokens(tokens)
    return Simhash(chunks).value


'''
def hash_text(text):
    """Return a stable hash (sha256) of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def tokenize_for_simhash(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    # Collapse whitespace
    tokens = text.split()
    return tokens

def simhash_text(text):
    tokens = tokenize_for_simhash(text)
    return Simhash(tokens).value
'''