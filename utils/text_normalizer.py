import os

_QUOTE_NORMALIZER = [
    ("\u201c", '"'), ("\u201d", '"'), ("\u201e", '"'), ("\u201f", '"'),
    ("\u2033", '"'), ("\u2036", '"'),
    ("\u2018", "'"), ("\u2019", "'"), ("\u201a", "'"), ("\u201b", "'"),
    ("\u2013", "-"), ("\u2014", "-"),
    ("\u2026", "..."),
]

DEBUG = os.getenv("DEBUG_NORMALIZER", "false").lower() == "true"


def normalize_text(text: str) -> str:
    if DEBUG:
        print("[DEBUG] normalize_text called")
    # Replace curly quotes, dashes, ellipses
    for bad, good in _QUOTE_NORMALIZER:
        text = text.replace(bad, good)
    # Final safeguard: strip unsupported characters
    if DEBUG:
        print("[DEBUG] Final Normalized Text: ",
              text.encode("utf-8", "ignore").decode("utf-8", "ignore"))
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")
