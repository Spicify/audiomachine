from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List, Tuple
from utils.text_normalizer import normalize_text as _norm_for_compare

try:
    import tiktoken
except Exception:  # Fallback if not installed yet
    tiktoken = None  # type: ignore


# Enhanced sentence boundary splitter:
# - Handles punctuation optionally followed by closing quotes
# - Handles ellipses (… or ...)
# - Handles em-dash (—) and en-dash (–)
# Each lookbehind alternative is fixed-width to remain compatible with Python's regex engine
_SENT_SPLIT_RE = re.compile(
    r'(?:(?<=\u2026)|(?<=\.\.\.)|(?<=[.!?]["”’\'])|(?<=[.!?])|(?<=\u2014)|(?<=\u2013))\s+'
)

# [DIAG] collect cross-speaker duplicate conflicts (normalized_text, speakerA, speakerB)
_DIAG_DEDUP_CONFLICTS: List[Tuple[str, str, str]] = []


def _split_into_sentences(text: str) -> List[str]:
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def _split_into_sentences_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """Split using the same boundary regex as _split_into_sentences, but return (text, start, end) spans.

    - start/end are character offsets in the original text for the trimmed sentence substring.
    - No behavior change to chunking; exported for diagnostics/ledger only.
    """
    if not text:
        return []
    spans: List[Tuple[str, int, int]] = []
    start = 0
    for m in _SENT_SPLIT_RE.finditer(text):
        end = m.start()
        raw = text[start:end]
        if raw:
            # trim and adjust indices accordingly
            ltrim = len(raw) - len(raw.lstrip())
            rtrim = len(raw.rstrip())
            seg = raw.strip()
            if seg:
                seg_start = start + ltrim
                seg_end = start + rtrim
                spans.append((seg, seg_start, seg_end))
        start = m.end()
    # tail
    if start <= len(text):
        raw = text[start:]
        if raw:
            ltrim = len(raw) - len(raw.lstrip())
            rtrim = len(raw.rstrip())
            seg = raw.strip()
            if seg:
                seg_start = start + ltrim
                seg_end = start + rtrim
                spans.append((seg, seg_start, seg_end))
    return spans


def _split_into_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n{2,}", text)
    return [p.strip() for p in parts if p and p.strip()]


def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _get_encoder(model: str | None = None):
    if tiktoken is None:
        return None
    try:
        if model:
            return tiktoken.encoding_for_model(model)
    except Exception:
        pass
    return tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    index: int
    text: str
    token_count: int
    prev_overlap_sentences: List[str]
    text_hash: str


def build_chunks(
    text: str,
    max_tokens: int = 3000,
    model: str | None = None,
    overlap_sentences: int = 2,
) -> List[Chunk]:
    """Token-aware chunking with sliding window sentence overlap.

    - Respects max token budget using tiktoken when available.
    - Uses paragraph → sentence segmentation for better boundaries.
    - Overlaps last N sentences between consecutive chunks (for continuity).
    """
    if not text or not text.strip():
        return []

    enc = _get_encoder(model)

    def tokens_len(s: str) -> int:
        if enc is None:
            return max(1, len(s) // 4)  # heuristic fallback
        return len(enc.encode(s))

    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: List[Chunk] = []
    current_sentences: List[str] = []

    def flush_chunk() -> None:
        nonlocal chunks, current_sentences
        if not current_sentences:
            return
        body = " ".join(current_sentences).strip()
        if body:
            prev_overlap = current_sentences[:overlap_sentences] if len(
                current_sentences) <= overlap_sentences else current_sentences[-overlap_sentences:]
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=body,
                    token_count=tokens_len(body),
                    prev_overlap_sentences=prev_overlap,
                    text_hash=_hash_text(body),
                )
            )
        current_sentences = []

    for para in paragraphs:
        sents = _split_into_sentences(para)
        for sent in sents:
            tentative = (" ".join(current_sentences + [sent])).strip()
            if tokens_len(tentative) <= max_tokens:
                current_sentences.append(sent)
            else:
                flush_chunk()
                # seed with overlap from previous chunk if any
                if chunks and overlap_sentences > 0:
                    last_overlap = chunks[-1].prev_overlap_sentences
                    current_sentences = last_overlap.copy()
                    # ensure we don't exceed immediately
                    seed = " ".join(current_sentences + [sent]).strip()
                    if tokens_len(seed) <= max_tokens:
                        current_sentences.append(sent)
                    else:
                        # sentence alone too big: hard split
                        current_sentences = [sent]
                else:
                    current_sentences = [sent]

        # paragraph boundary: prefer flush if large
        if tokens_len(" ".join(current_sentences)) >= max_tokens * 0.8:
            flush_chunk()

    flush_chunk()
    return chunks


def deduplicate_lines(lines: List[dict]) -> List[dict]:
    """Deduplicate JSONL output lines.

    Primary: exact (character|text) hash.
    Secondary: cross-character normalized text match → prefer non-Narrator.
    """
    def _norm_text(s: str) -> str:
        # Delegate to shared normalizer for cross-module consistency
        return _norm_for_compare(s or "")

    seen = set()
    result: List[dict] = []
    norm_to_index: dict[str, int] = {}

    for idx, it in enumerate(lines):
        key = _hash_text(
            f"{(it.get('character') or '').strip().lower()}|{(it.get('text') or '').strip()}")
        if key in seen:
            try:
                print(
                    f"[DEDUP_DECISION] action=drop sim=1.000 reason=exact idx={idx} txt='{(it.get('text','') or '')[:60]}'", flush=True)
            except Exception:
                pass
            continue
        txt_norm = _norm_text(it.get("text", ""))
        if txt_norm:
            if txt_norm in norm_to_index:
                # prefer non-Narrator version
                existing_idx = norm_to_index[txt_norm]
                existing = result[existing_idx]
                # Short fresh insertion guard for FB/REINJ content
                try:
                    a_txt = (existing or {}).get("text", "")
                    b_txt = (it or {}).get("text", "")
                    a_n = _norm_for_compare(a_txt)
                    b_n = _norm_for_compare(b_txt)
                    short_guard = (min(len(a_n), len(b_n)) <= 25)
                    fresh_src = (((it or {}).get("_src") in {"fb", "reinj"}) or (
                        (existing or {}).get("_src") in {"fb", "reinj"}))
                    if short_guard and fresh_src:
                        # Only drop if truly identical after normalization
                        if a_n != b_n:
                            try:
                                print(
                                    f"[DEDUP_DECISION] skip_drop short_guard src={(it or {}).get('_src')} text='{b_n[:60]}'", flush=True)
                            except Exception:
                                pass
                            seen.add(key)
                            result.append(it)
                            if txt_norm and txt_norm not in norm_to_index:
                                norm_to_index[txt_norm] = len(result) - 1
                            continue
                except Exception:
                    pass
                cur_is_narr = str(it.get("character", "")
                                  ).strip() == "Narrator"
                exist_is_narr = str(existing.get(
                    "character", "")).strip() == "Narrator"
                if exist_is_narr and not cur_is_narr:
                    # replace existing narrator with character version
                    result[existing_idx] = it
                    try:
                        print(
                            f"[DEDUP_PREF] replaced Narrator with {it.get('character')} for '{txt_norm[:60]}'",
                            flush=True,
                        )
                    except Exception:
                        pass
                    seen.add(key)
                    continue
                # else, drop current narrator duplicate
                if cur_is_narr:
                    try:
                        print(
                            f"[DEDUP_DECISION] action=drop sim=1.000 reason=normalized-narrator idx={idx} txt='{(it.get('text','') or '')[:60]}'", flush=True)
                    except Exception:
                        pass
                    continue
                # [DIAG] conflict: same normalized text with two non-narrator speakers
                try:
                    if not cur_is_narr and not exist_is_narr:
                        _a = str(existing.get("character", ""))
                        _b = str(it.get("character", ""))
                        print(
                            f"[DIAG] Cross-speaker duplicate conflict: '{txt_norm[:80]}' speakers=({_a} vs {_b})", flush=True)
                        try:
                            _DIAG_DEDUP_CONFLICTS.append((txt_norm, _a, _b))
                        except Exception:
                            pass
                        # Preference rule: prefer non-'Ambiguous' over 'Ambiguous'
                        if _a == "Ambiguous" and _b != "Ambiguous":
                            result[existing_idx] = it
                            continue
                        if _b == "Ambiguous" and _a != "Ambiguous":
                            # keep existing
                            continue
                        # If both named but different, keep existing (first wins) and drop current
                        # (we could enhance with last_speaker if passed here)
                        continue
                except Exception:
                    pass
        seen.add(key)
        result.append(it)
        if txt_norm and txt_norm not in norm_to_index:
            norm_to_index[txt_norm] = len(result) - 1
    return result


def diag_consume_dedup_conflicts(max_items: int | None = None) -> List[Tuple[str, str, str]]:
    """Return and clear collected dedup conflicts for DIAG summaries."""
    try:
        out = list(_DIAG_DEDUP_CONFLICTS)
        _DIAG_DEDUP_CONFLICTS.clear()
        if max_items is not None:
            return out[:max_items]
        return out
    except Exception:
        return []


def deduplicate_lines_exact(lines: list) -> list:
    """
    Legacy, conservative de-dup:
    - Build a key from (character, normalized text)
    - Keep first occurrence, drop exact/near-exact repeats only
    - No speaker preference heuristics
    """
    import re
    from difflib import SequenceMatcher

    def _norm(s: str) -> str:
        return _norm_for_compare(s or "")

    seen = []
    out = []
    for idx, obj in enumerate(lines):
        ch = (obj or {}).get("character", "")
        tx = (obj or {}).get("text", "")
        key = (ch.strip().lower(), _norm(tx))
        # exact seen?
        if key in seen:
            try:
                print(
                    f"[DEDUP_DECISION] action=drop sim=1.000 reason=exact idx={idx} txt='{(tx or '')[:60]}'", flush=True)
            except Exception:
                pass
            continue
        # light near-dup guard: if any seen text is ~identical, drop
        is_near_dup = False
        for sch, stx in seen:
            if sch != key[0]:
                continue
            # Short fresh insertion guard for short FB/REINJ lines before dropping
            try:
                a_n = stx
                b_n = key[1]
                short_guard = (min(len(a_n), len(b_n)) <= 25)
                fresh_src = ((obj or {}).get("_src") in {"fb", "reinj"})
                if short_guard and fresh_src and (a_n != b_n):
                    try:
                        print(
                            f"[DEDUP_DECISION] skip_drop short_guard src={(obj or {}).get('_src')} text='{b_n[:60]}'", flush=True)
                    except Exception:
                        pass
                    continue
            except Exception:
                pass
            if SequenceMatcher(None, key[1], stx).ratio() >= 0.98:
                is_near_dup = True
                break
        if is_near_dup:
            try:
                print(
                    f"[DEDUP_DECISION] action=drop sim=0.980 reason=near-exact idx={idx} txt='{(tx or '')[:60]}'", flush=True)
            except Exception:
                pass
            continue
        seen.append(key)
        out.append(obj)
    return out
