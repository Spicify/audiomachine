from __future__ import annotations

from typing import Any, Dict, List
import re
import difflib

from difflib import SequenceMatcher as _SM
from utils.text_normalizer import normalize_text as _norm_for_compare


def _extract_quotes(text: str) -> str:
    """Extract spoken parts inside quotes for fair comparison against Friendli text."""
    try:
        quotes = re.findall(r'“([^"]+)?"|"([^"]+)?"', text or "")
        if quotes:
            return " ".join([(q[0] or q[1]) for q in quotes])
    except Exception:
        pass
    return text or ""


def _dedupe_chunk_boundaries(all_chunks: List[List[Dict[str, Any]]], threshold: float = 0.9) -> List[Dict[str, Any]]:
    """
    Compare only the last line of each chunk with the first line of the next one.
    If similarity ≥ threshold, drop the earlier line (keep later chunk's version).
    Works safely with 1+ chunks.
    """
    if not all_chunks or len(all_chunks) <= 1:
        return [line for chunk in (all_chunks or []) for line in chunk]

    deduped: List[List[Dict[str, Any]]] = []
    for i, chunk in enumerate(all_chunks):
        if not chunk:
            continue
        if i == 0:
            deduped.append(chunk)
            continue

        prev_chunk = deduped[-1] if deduped else []
        current_chunk = chunk

        if not prev_chunk or not current_chunk:
            deduped.append(current_chunk)
            continue

        last_prev = prev_chunk[-1]
        first_curr = current_chunk[0]

        sim = _SM(None, (last_prev.get("text", "") or ""),
                  (first_curr.get("text", "") or "")).ratio()
        if sim >= threshold:
            try:
                print(
                    f"[DEDUP] Dropping overlapping line (sim={sim:.2f}): {((last_prev.get('text','') or '')[:80])!r}", flush=True)
            except Exception:
                pass
            prev_chunk = prev_chunk[:-1]

        if deduped:
            deduped[-1] = prev_chunk
        deduped.append(current_chunk)

    flat: List[Dict[str, Any]] = [line for chunk in deduped for line in chunk]
    try:
        print(
            f"[EOD][DEDUP_SUMMARY] merged_chunks={len(all_chunks)} → after_dedup={len(flat)} lines", flush=True)
    except Exception:
        pass
    return flat


def _normalize_text_for_match(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u201C", '"').replace("\u201D", '"')
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _build_sentence_to_pos_map(chunk_text: str, dialogues: list) -> dict:
    """
    Map sentence index → earliest dialogue index that best covers that sentence.
    We consider containment or high similarity against dialogue text.
    """
    def _split_sentences_robust(text: str) -> list:
        t = (text or "")
        t = t.replace("\u201C", '"').replace("\u201D", '"')
        t = t.replace("\u2018", "'").replace("\u2019", "'")
        t = t.replace("\u2026", "...")
        boundary = re.compile(
            r'(?<=[.!?])\s+'
            r'|(?<=,")\s+'
            r'|(?<=,)\s+'
            r'|(?<=\")\s+(?=(?:[A-Z]|he|she|they))'
            r'|(?<=\.)\s+(?=\")'
        )
        parts = [p.strip() for p in boundary.split(t) if p and p.strip()]
        return parts

    sents = _split_sentences_robust(chunk_text)
    sent_norms = [_norm_for_compare(s) for s in sents]

    pos_map = {}
    for di, obj in enumerate(dialogues):
        txt = _norm_for_compare((obj or {}).get("text", ""))
        if not txt:
            continue
        best_idx, best_sim = None, 0.0
        for si, sn in enumerate(sent_norms):
            if not sn:
                continue
            if sn and (sn in txt or txt in sn):
                best_idx, best_sim = si, 1.0
                break
            sim = difflib.SequenceMatcher(None, sn, txt).ratio()
            if sim > best_sim:
                best_idx, best_sim = si, sim
        if best_idx is not None:
            pos_map.setdefault(best_idx, di)
    return pos_map


def _simple_reinject_missing_as_narrator(original_text: str, lines: list) -> list:
    """
    Legacy behavior: ensure every input sentence is represented.
    - Split original_text into simple sentences.
    - Normalize and compare against produced lines' text.
    - For any sentence not covered by any line, append a Narrator line with that exact sentence.
    - Preserve input order for any injected lines by scanning in input sequence and appending in that order.
    """
    import re as _re
    from difflib import SequenceMatcher

    def _split_sentences_robust(text: str) -> list[str]:
        """
        Robust sentence splitter for reinjection:
        - Treats standard sentence endings (. ! ?) as boundaries.
        - Also treats comma+closing-quote (,") or ,’) as a boundary commonly found in dialogue.
        - Handles curly quotes, ellipsis and em dashes.
        - Avoids over-splitting on simple commas.
        """
        import re as __re

        if not text:
            return []

        t = text
        t = t.replace("\u201C", '"').replace("\u201D", '"')
        t = t.replace("\u2018", "'").replace("\u2019", "'")
        t = t.replace("\u2026", "...")

        boundary = __re.compile(
            r'(?<=[.!?])\s+'                  # sentence end
            r'|(?<=,")\s+'                    # comma followed by closing quote
            # r'|(?<=,)\s+'                   # ❌ removed: plain comma split caused attribution tails
            r'|(?<=\")\s+(?=(?:[A-Z]|he|she|they))'
            r'|(?<=\.)\s+(?=\")'
        )

        parts = [p.strip() for p in boundary.split(t) if p and p.strip()]
        return parts

    def _normalize(s: str) -> str:
        s = (s or "")
        # unify quotes/apostrophes/ellipsis
        s = s.replace("\u201C", '"').replace("\u201D", '"')
        s = s.replace("\u2018", "'").replace("\u2019", "'")
        s = s.replace("\u2026", "...")
        # lowercase + collapse
        s = _re.sub(r"\s+", " ", s).strip().lower()
        # strip terminal punctuation (tolerate comma/period differences)
        s = s.rstrip('.,;:!?"\'')
        return s

    raw_sents = _split_sentences_robust(original_text)
    raw_sents = [s for s in raw_sents if s.strip()]

    produced_texts_norm = []
    for obj in lines:
        try:
            txt = (obj.get("text", "") or "")
            txt = txt.replace("\u201C", '"').replace(
                "\u201D", '"').replace("\u2018", "'").replace("\u2019", "'")
            produced_texts_norm.append(_normalize(txt))
        except Exception:
            produced_texts_norm.append("")

    def _covered(sent_norm: str) -> bool:
        for pt in produced_texts_norm:
            if not pt:
                continue
            if sent_norm in pt or pt in sent_norm:
                return True
            if SequenceMatcher(None, sent_norm, pt).ratio() >= 0.92:
                return True
        return False

    reinjected = []

    # Pre-compute quoted spans from original_text for coverage decisions
    try:
        _q = re.findall(r'“([^”]+)”|"([^"]+)"', original_text or "")
        _quoted_norm = {_normalize((a or b or ""))
                        for (a, b) in _q if (a or b)}
    except Exception:
        _quoted_norm = set()

    # broadened attribution-only detector to align with streaming
    def _is_attrib_only_tail(txt: str) -> bool:
        import re as __re
        if not txt:
            return False
        if ('"' in txt) or ('\u201C' in txt) or ('\u201D' in txt):
            return False
        t2 = txt.strip()
        if not __re.match(r'^(?:[A-Z][a-z]+|[A-Z][A-Za-z\-]+|he|she|they)\b', t2, flags=__re.IGNORECASE):
            return False
        _ATTRIB_VERBS2 = r"(?:said|asked|replied|answered|added|remarked|observed|noted|stated|whispered|murmured|muttered|mumbled|stammered|stuttered|shouted|yelled|bellowed|thundered|exclaimed|snapped|barked|spat|warned|cried|sobbed|wailed|whimpered|gasped|moaned|groaned|panted|hissed|growled|snarled|grunted|pleaded|begged|implored|commanded|demanded|ordered|laughed|chuckled|giggled|snorted|teased|taunted|sighed|purred)"
        return bool(__re.search(rf"\b{_ATTRIB_VERBS2}\b", t2, flags=__re.IGNORECASE))

    for s in raw_sents:
        sn = _normalize(s)
        if not _covered(sn):
            # Skip pure attribution tails if the quoted content is already covered
            if _is_attrib_only_tail(s):
                if _quoted_norm and any((_normalize(q) in produced_texts_norm) for q in _quoted_norm):
                    try:
                        print("[REINJ_SKIP] reason=attrib_tail_covered", flush=True)
                    except Exception:
                        pass
                    continue

            def _is_raw_quote(s: str) -> bool:
                t = (s or "")
                return ('"' in t) or ('\u201C' in t) or ('\u201D' in t)
            if _is_raw_quote(s):
                snq = _normalize(s).rstrip('.,;:!?"\'')
                # if this sentence is essentially just a quoted span that is already present in outputs, skip
                if snq in _quoted_norm and any(snq == pt for pt in produced_texts_norm):
                    try:
                        print(
                            "[REINJ_SKIP] reason=raw_quote_already_covered", flush=True)
                    except Exception:
                        pass
                    continue
            reinjected.append({
                "character": "Narrator",
                "emotions": [],
                "text": s,
                "_src": "reinj"
            })
    if not reinjected:
        return lines

    return lines + reinjected


def _preclean_jsonl(raw: str) -> str:
    txt = (raw or "").strip()
    if not txt:
        return ""
    try:
        if (txt.startswith("```") and txt.endswith("```") or (txt.startswith("~~~") and txt.endswith("~~~"))):
            txt = txt.strip("`~\n ")
    except Exception:
        pass
    try:
        import json as _json
        import re as _re
        if re.match(r"^\s*\[", txt):
            arr = _json.loads(txt)
            if isinstance(arr, list):
                return "\n".join(_json.dumps(obj, ensure_ascii=False) for obj in arr if isinstance(obj, dict))
    except Exception:
        pass
    try:
        import re as _re
        blocks = _re.findall(r"\{[\s\S]*?\}", txt, _re.DOTALL)
        if blocks:
            cleaned_lines = []
            for b in blocks:
                b2 = _re.sub(r",\s*$", "", b.strip(), flags=_re.MULTILINE)
                cleaned_lines.append(b2)
            return "\n".join(cleaned_lines)
    except Exception:
        pass
    return txt
