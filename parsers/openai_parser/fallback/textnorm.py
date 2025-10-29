import re
from utils.text_normalizer import normalize_text as _norm_for_compare


_QUOTE_CHARS = "\"'“”‘’«»‹›`´˝˝ˮ‟❝❞〝〞″‶″❮❯"


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = s.translate(str.maketrans({
        "“": '"', "”": '"', "‘": "'", "’": "'",
    }))
    s = s.strip(_QUOTE_CHARS)
    s = s.replace("—", " ").replace("–", "-")
    s = _collapse_ws(s)
    return s.lower()


def _norm_for_compare_punct_neutral(s: str) -> str:
    s = _norm_for_compare(s or "")
    s = _collapse_ws(s).lower()
    s = s.rstrip('.,;:!?"\'')
    return s


def _pnorm(s: str) -> str:
    s = _norm_for_compare_punct_neutral(s or "")
    return re.sub(r"\s+", " ", s).strip().lower()


def _tok_sim(a: str, b: str) -> float:
    at, bt = set((a or "").split()), set((b or "").split())
    return (len(at & bt) / len(at | bt)) if (at and bt) else 0.0


def _edit_ratio(a: str, b: str) -> float:
    try:
        import difflib as _df
        return _df.SequenceMatcher(None, a, b).ratio()
    except Exception:
        return 0.0
