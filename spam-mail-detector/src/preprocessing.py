import re

# Keep preprocessing deliberately lightweight. TF-IDF handles tokenization
# and vocabulary construction, while this function removes noisy formatting.
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = re.sub(r"\d+", " numbertoken ", text)
    text = re.sub(r"[^a-z0-9_\s]", " ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text
