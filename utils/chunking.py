import re
from typing import List, Dict


def _split_into_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n{2,}", text)
    return [p.strip() for p in parts if p and p.strip()]


def _split_into_sentences(paragraph: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [s.strip() for s in sentences if s and s.strip()]


def chunk_text(project_id: str, text: str, max_chars: int = 2000) -> List[Dict[str, str]]:
    if not text:
        return []

    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if len(para) <= max_chars and (len(current) + len(para) + (2 if current else 0)) <= max_chars:
            current = f"{current}\n\n{para}" if current else para
            continue

        sentences = _split_into_sentences(para)
        for sent in sentences:
            if len(sent) > max_chars:
                start = 0
                while start < len(sent):
                    piece = sent[start:start + max_chars]
                    if current:
                        chunks.append(current)
                        current = ""
                    chunks.append(piece)
                    start += max_chars
                continue

            if not current:
                current = sent
            elif len(current) + 1 + len(sent) <= max_chars:
                current = f"{current} {sent}"
            else:
                chunks.append(current)
                current = sent

        if current and len(current) + 2 <= max_chars:
            current = f"{current}\n\n"

    if current and current.strip():
        chunks.append(current.strip())

    width = max(3, len(str(len(chunks))))
    result: List[Dict[str, str]] = []
    for i, c in enumerate(chunks, start=1):
        cid = f"{project_id}_chunk{str(i).zfill(width)}"
        result.append({"id": cid, "text": c})

    return result
