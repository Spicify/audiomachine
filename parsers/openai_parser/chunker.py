from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List, Tuple

try:
    import tiktoken
except Exception:  # Fallback if not installed yet
    tiktoken = None  # type: ignore


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_into_sentences(text: str) -> List[str]:
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p and p.strip()]


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
    """Deduplicate JSONL output lines based on (character, text) hash.

    Used after overlapping chunk inference to remove repeated lines.
    """
    seen = set()
    result: List[dict] = []
    for it in lines:
        key = _hash_text(
            f"{(it.get('character') or '').strip().lower()}|{(it.get('text') or '').strip()}")
        if key in seen:
            continue
        seen.add(key)
        result.append(it)
    return result
