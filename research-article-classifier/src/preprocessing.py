import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

EXTRA_WORDS = [
    "abstract", "background", "method", "methods", "result", "results",
    "conclusion", "conclusions", "objective", "aim", "purpose", "introduction",
    "paper", "study", "article", "research", "elsevier", "springer", "wiley",
    "copyright", "rights", "reserved", "author", "authors", "published", "license",
    "social", "business", "science",
]
STOP_WORDS = set(ENGLISH_STOP_WORDS) | set(EXTRA_WORDS)

def make_singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and not word.endswith("us"):
        return word[:-1]
    return word

def clean_text(text) -> str:
    """Apply the same text cleaning used in the training notebook."""
    text = str(text).split("©")[0]
    text = text.lower()
    text = re.sub(r"[^a-z ]", " ", text)
    words = text.split()
    cleaned_words = []
    for word in words:
        word = make_singular(word)
        if word not in STOP_WORDS and len(word) > 2:
            cleaned_words.append(word)
    return " ".join(cleaned_words)
