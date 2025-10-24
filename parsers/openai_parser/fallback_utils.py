import os
from openai import OpenAI
import time
import re
from dotenv import load_dotenv
from utils.text_normalizer import normalize_text as _norm_for_compare

# Ensure .env is loaded before reading environment variables
load_dotenv()
print("[DEBUG] .env loaded for Frendli fallback")


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _build_fallback_system_prompt(known_characters: list[str] | None = None) -> str:
    """Concise strict JSONL system prompt for fallback (≈25 lines)."""
    chars_str = ", ".join(known_characters or [])
    lines = [
        "You are a strict audiobook dialogue parser.",
        "Output MUST be JSON Lines (JSONL). One JSON object per line with keys: character, emotions (2 strings), text. Optional: candidates (2–5 strings) only when character == 'Ambiguous'.",
        "No extra text, no commentary, no blank lines.",
        "Do NOT invent new character names under any circumstances.",
        "If the speaker is unclear, use 'Ambiguous' and choose candidates ONLY from Known characters.",
        "Coverage: Do NOT skip any input sentence.",
        "If speaker unclear → use 'Ambiguous' with 2–5 candidates from known characters.",
        "Narration: Use 'Narrator' for objective, third-person description (not tied to any POV).",
        "Quotes: The text field must contain ONLY the spoken words inside quotes (drop attributions like 'he growled').",
        "When quotes include attribution verbs (e.g., said, asked, whispered, commanded, moaned, murmured, replied), exclude those verbs from the 'text' (keep only the words inside quotes).",
        "Speaker inference: When a quoted dialogue appears after a character’s name or pronoun (e.g., Mark said, '…' or he whispered, '…'), infer that character as the speaker of the quoted text.",
        "Emotions: Exactly TWO per line.",
        f"Known characters: [{chars_str}]",
        "Examples:",
        '{"character":"Narrator","emotions":["soft","sad"],"text":"The rain poured outside."}',
        '{"character":"Bella","emotions":["confused","tense"],"text":"I can’t believe he said that."}',
        '{"character":"Ambiguous","emotions":["curious","soft"],"candidates":["Aria","Luca"],"text":"You two need to keep quiet."}',
        '{"character":"Luca","emotions":["angry","tense"],"text":"Get up!"}',
        # New concise example for attribution verbs inside quotes
        'Input: "Keep your eyes on me," she commanded. → Output: {"character":"Maya","emotions":["commanding","dominant"],"text":"Keep your eyes on me."}',
    ]
    return "\n".join(lines)


def call_frendli_fallback(system_prompt: str, user_prompt: str, known_characters: list[str] | None = None) -> str:
    """
    Calls Frendli serverless endpoint for fallback parsing.
    Returns the raw JSONL string.
    """
    token_ok = os.getenv("FRIENDLI_TOKEN") is not None
    print(f"[DEBUG] Frendli token loaded: {token_ok}", flush=True)
    print("[DEBUG] FRIENDLI_TOKEN found:", bool(os.getenv("FRIENDLI_TOKEN")))

    client = OpenAI(
        api_key=os.getenv("FRIENDLI_TOKEN"),
        base_url="https://api.friendli.ai/serverless/v1",
    )

    # Verify client URL/debug
    try:
        print("[DEBUG] Friendli client base_url:",
              getattr(client, "base_url", None))
        print("[DEBUG] Frendli request URL verified", flush=True)
    except Exception:
        pass

    # Construct combined system prompt (strict JSONL rules)
    strict_system_prompt = _build_fallback_system_prompt(known_characters)
    # [DIAG] show first lines of system prompt
    try:
        print("[DIAG] Friendli system prompt (first 12 lines):", flush=True)
        for i, ln in enumerate(strict_system_prompt.splitlines()[:12], 1):
            print(f"[DIAG]   {i:02d}: {ln}", flush=True)
        _kc_empty = (not known_characters) or (isinstance(
            known_characters, list) and len(known_characters) == 0)
        _has_dont_invent = ("do not invent" in strict_system_prompt.lower())
        print(
            f"[DIAG] Known characters empty: {_kc_empty}; Has 'Do NOT invent names' clause: {_has_dont_invent}", flush=True)
    except Exception:
        pass

    start = time.monotonic()
    print("[DEBUG] Frendli request begin", flush=True)
    try:
        response = client.chat.completions.create(
            model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            messages=[
                {"role": "system", "content": strict_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        out = response.choices[0].message.content if response and getattr(
            response, "choices", None) else ""
        elapsed = (time.monotonic() - start)
        print(
            f"[DEBUG] Frendli request end ({elapsed*1000.0:.0f} ms, chars={len(out)})", flush=True)
        return out
    except Exception as e:
        print(f"[ERROR] Frendli request failed: {repr(e)}", flush=True)
        raise


_QUOTE_CHARS = "\"'“”‘’«»‹›`´˝˝ˮ‟❝❞〝〞″‶″❮❯"


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
    # existing helper; handles curly quotes/ellipsis
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


def _extract_quotes(text: str) -> list[str]:
    """Extract normalized quoted spans. Never raises; returns [] if none.

    - Handles straight and curly quotes
    - Trims whitespace and trailing punctuation
    - Logs a brief diagnostic
    """
    out: list[str] = []
    try:
        if not text:
            print("[FB_QOUTES] extracted=0 samples=[]", flush=True)
            return out
        import re as _re
        raw = _re.findall(r'“([^”]+)”|"([^"]+)"', text)
        for g1, g2 in raw:
            q = (g1 or g2 or "").strip()
            if not q:
                continue
            q = _normalize_text(q)
            # strip terminal punctuation
            q = q.rstrip('.,;:!?"\'')
            if q:
                out.append(q)
    except Exception:
        out = []
    try:
        _samples = out[:2]
        print(
            f"[FB_QOUTES] extracted={len(out)} samples={_samples}", flush=True)
    except Exception:
        pass
    return out


# Shared normalizer is imported as _norm_for_compare


def _fb_is_attrib_only(text: str) -> bool:
    if not text:
        return False
    if ('"' in text) or ('\u201C' in text) or ('\u201D' in text):
        return False
    t = text.strip()
    if not re.match(r'^(?:[A-Z][a-z]+|[A-Z][A-Za-z\-]+|he|she|they)\b', t, flags=re.IGNORECASE):
        return False
    _ATTRIB_VERBS = r"(?:said|asked|replied|answered|added|remarked|observed|noted|stated|whispered|murmured|muttered|mumbled|stammered|stuttered|shouted|yelled|bellowed|thundered|exclaimed|snapped|barked|spat|warned|cried|sobbed|wailed|whimpered|gasped|moaned|groaned|panted|hissed|growled|snarled|grunted|pleaded|begged|implored|commanded|demanded|ordered|laughed|chuckled|giggled|snorted|teased|taunted|sighed|purred)"
    return bool(re.search(rf"\b{_ATTRIB_VERBS}\b", t, flags=re.IGNORECASE))


def _token_jaccard(a: str, b: str) -> float:
    at = set((a or "").split())
    bt = set((b or "").split())
    if not at or not bt:
        return 0.0
    inter = len(at & bt)
    union = len(at | bt)
    return (inter / union) if union else 0.0


def _edit_ratio(a: str, b: str) -> float:
    try:
        import difflib as _df
        return _df.SequenceMatcher(None, a, b).ratio()
    except Exception:
        return 0.0


def _sanitize_character(line: dict, known_characters: list[str] | None) -> dict:
    ch = str((line or {}).get("character", "")).strip()
    allowed = set((known_characters or [])) | {"Narrator", "Ambiguous"}
    if ch and ch not in allowed:
        line["character"] = "Ambiguous"
        # keep candidates small; optional
        if not line.get("candidates"):
            line["candidates"] = list((known_characters or [])[:5])
    return line


def filter_fallback_lines(segment_text: str, candidate_lines: list[dict]) -> list[dict]:
    """Filter Friendli candidate lines against the specific problem segment text.

    - Narrator: keep if contained in segment or token-similarity ≥ 0.30
    - Dialogue with quotes: keep if matches any quoted span or token-similarity ≥ 0.50 against a quoted span
    - Dialogue without quotes: keep if contained in segment or token-similarity ≥ 0.45
    Logs [FB_FUZZY_KEEP]/[FB_FUZZY_DROP] per-candidate.
    """
    try:
        seg_text = (segment_text or "")
        seg_norm = _pnorm(seg_text)
        quoted_in_seg = set(_pnorm(q) for q in _extract_quotes(seg_text))

        kept: list[dict] = []
        for ln in (candidate_lines or []):
            try:
                cand = _pnorm((ln or {}).get("text", ""))
                ch = (ln or {}).get("character", "") or ""
                keep = False
                reason = ""

                if ch == "Narrator":
                    if (cand and (cand in seg_norm or seg_norm.find(cand) >= 0)) or _tok_sim(cand, seg_norm) >= 0.30:
                        keep, reason = True, "narr_contained_or_sim>=0.30"
                else:
                    if quoted_in_seg:
                        if cand in quoted_in_seg or any(_tok_sim(cand, q) >= 0.50 for q in quoted_in_seg):
                            keep, reason = True, "quoted_match"
                    else:
                        if (cand and (cand in seg_norm or seg_norm.find(cand) >= 0)) or _tok_sim(cand, seg_norm) >= 0.45:
                            keep, reason = True, "no_quotes_seg_match"

                if keep:
                    try:
                        print(
                            f"[FB_FUZZY_KEEP] ch={ch} reason={reason} text='{(ln.get('text','') or '')[:80]}'", flush=True)
                    except Exception:
                        pass
                    kept.append(ln)
                else:
                    try:
                        print(
                            f"[FB_FUZZY_DROP] ch={ch} text='{(ln.get('text','') or '')[:80]}'", flush=True)
                    except Exception:
                        pass
            except Exception:
                # on per-candidate error, drop silently
                pass

        return kept

    except Exception as e:
        import traceback as _tb
        cause = f"{type(e).__name__}: {e}"
        here = _tb.extract_tb(e.__traceback__)[-1] if e.__traceback__ else None
        loc = f"{getattr(here, 'filename', 'unknown')}:{getattr(here, 'lineno', '?')}"
        try:
            print(
                f"[EXC_TRACE] stage=fallback_validation error='{cause}' at={loc}", flush=True)
        except Exception:
            pass
        try:
            from .openai_parser import DEBUG_PARSER_DIAG as _DBG
        except Exception:
            _DBG = False
        if _DBG:
            raise
        return list(candidate_lines or [])


_SENT_SPLIT_RX = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> list[dict]:
    parts: list[dict] = []
    if not text:
        return parts
    # Normalize whitespace without altering content semantics
    norm_text = _collapse_ws(text)
    chunks = re.split(_SENT_SPLIT_RX, norm_text)
    idx = 0
    for c in chunks:
        c = (c or "").strip()
        if not c:
            continue
        norm = _norm_for_compare_punct_neutral(c)
        if norm and norm not in {'"', "'"}:
            parts.append({"index": idx, "text": c, "norm": norm})
            idx += 1
    return parts


def _group_consecutive(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    indices = sorted(set(indices))
    groups = []
    start = prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        groups.append((start, prev))
        start = prev = i
    groups.append((start, prev))
    return groups


def detect_missing_or_rejected_lines(chunk_text, parsed_lines):
    """
    Detects missing or rejected lines from OpenAI output.
    - Finds explicit REJECTED tags.
    - Compares chunk sentences to parsed output.
    """
    sent_infos = _split_sentences(chunk_text)
    parsed_norm_texts = [_norm_for_compare_punct_neutral(
        d.get("text", "")) for d in parsed_lines]

    uncovered_indices: list[int] = []
    for si in sent_infos:
        norm_sent = si["norm"]
        if not norm_sent:
            continue
        # Baseline coverage by containment (neutral)
        covered = any(
            norm_sent in pt or pt in norm_sent for pt in parsed_norm_texts if pt)

        # QUOTE-AWARE coverage: if the sentence has a quoted core that matches any parsed text, consider it covered
        if not covered:
            quoted_spans = _extract_quotes(si["text"])
            if quoted_spans:
                quoted_norms = [_norm_for_compare_punct_neutral(
                    q) for q in quoted_spans]
                if any(qn and any(qn == pt for pt in parsed_norm_texts) for qn in quoted_norms):
                    covered = True
        # --- NEW: token-similarity fallback detection for partial misses ---
        if not covered:
            def _token_similarity(a: str, b: str) -> float:
                atoks, btoks = set((a or "").split()), set((b or "").split())
                if not atoks or not btoks:
                    return 0.0
                inter = len(atoks & btoks)
                union = len(atoks | btoks)
                if union == 0:
                    return 0.0
                return inter / union

            for pt in parsed_norm_texts:
                if not pt:
                    continue
                sim = _token_similarity(norm_sent, pt)
                if sim >= 0.6:
                    covered = True
                    break
            if not covered:
                try:
                    print(
                        f"[DIAG][SIM] missing segment detected by token similarity (no match ≥0.6): '{norm_sent[:80]}'", flush=True)
                except Exception:
                    pass
        if not covered:
            uncovered_indices.append(si["index"])

    missing_groups = _group_consecutive(uncovered_indices)
    missing_segments = []
    for start_idx, end_idx in missing_groups:
        joined_text = "\n".join(
            si["text"] for si in sent_infos if start_idx <= si["index"] <= end_idx)
        missing_segments.append({
            "start_idx": start_idx,
            "end_idx": end_idx,
            "text": joined_text,
            "kind": "missing",
        })

    rejected_segments = []
    for l in parsed_lines:
        if l.get("character") == "REJECTED":
            rtxt = l.get("text") or ""
            rnorm = _normalize_text(rtxt)
            best_i = None
            for si in sent_infos:
                sn = si["norm"]
                if not sn:
                    continue
                if rnorm in sn or sn in rnorm:
                    best_i = si["index"]
                    break
            if best_i is not None:
                rejected_segments.append({
                    "start_idx": best_i,
                    "end_idx": best_i,
                    "text": rtxt,
                    "kind": "rejected",
                })

    # --- SAFETY NET: Always ensure at least one rejected segment exists if OpenAI emitted REJECTED lines ---
    try:
        if any(l.get("character") == "REJECTED" for l in parsed_lines):
            if not rejected_segments:
                seg_text = (chunk_text or "").strip(
                ) or "REJECTED_SEGMENT_FALLBACK"
                rejected_segments.append({
                    "start_idx": 0,
                    "end_idx": max(len(sent_infos) - 1, 0),
                    "text": seg_text,
                    "kind": "rejected",
                })
                print(
                    f"[DIAG] SAFETY-NET: Added fallback segment for unmapped REJECTED line(s) "
                    f"(len(sent_infos)={len(sent_infos)} text_len={len(seg_text)})",
                    flush=True,
                )
    except Exception:
        pass

    all_problem_lines = sorted(
        missing_segments + rejected_segments, key=lambda x: x["start_idx"])
    # Filter out overlaps where the segment text already appears in parsed lines (normalized)
    skip = 0
    kept_rejected = 0
    filtered: list[dict] = []
    try:
        print(
            "[DIAG] detect_missing_or_rejected_lines() received",
            len(parsed_lines),
            "lines with",
            sum(1 for l in parsed_lines if str(
                l.get("character", "")).upper() == "REJECTED"),
            "REJECTED",
            flush=True,
        )
    except Exception:
        pass
    for seg in all_problem_lines:
        # Always keep REJECTED segments
        if str(seg.get("kind", "")).lower() == "rejected":
            filtered.append(seg)
            kept_rejected += 1
            continue
        seg_norm = _normalize_text(seg.get("text", ""))
        if any(seg_norm in pt or pt in seg_norm for pt in parsed_norm_texts if pt):
            skip += 1
            continue
        filtered.append(seg)
    try:
        print(
            f"[DEBUG] Detection: missing_groups={len(missing_segments)} rejected_mapped={len(rejected_segments)} (total={len(all_problem_lines)})",
            flush=True,
        )
        print(
            f"[DEBUG] Fallback overlap filtered: {skip} skipped (REJECTED kept {kept_rejected})", flush=True)
    except Exception:
        pass
    return filtered


def replace_or_insert_lines(dialogues, new_lines, start_index=None, end_index=None, logger=None):
    """
    Insert or replace lines with priority:
    1) Per-line `_span_start` (deterministic).
    2) Group-level positional guess (start_index/end_index clamped).
    3) Fuzzy content anchor near the guessed position.
    4) Final fallback: tail-append.
    """
    import re
    from difflib import SequenceMatcher
    try:
        print(f"[REINJ_ARGS] new_lines_t={type(new_lines).__name__} new_len={len(new_lines) if isinstance(new_lines, list) else -1} start_t={type(start_index).__name__} end_t={type(end_index).__name__}", flush=True)
    except Exception:
        pass

    def _n(s: str) -> str:
        if not s:
            return ""
        s = s.replace("\u201C", '"').replace("\u201D", '"')
        s = s.replace("\u2018", "'").replace("\u2019", "'")
        s = _collapse_ws(s).lower()
        return s

    def _clamp(i: int, lo: int, hi: int) -> int:
        return max(lo, min(i, hi))

    # 1) If ALL new_lines carry _span_start → deterministic inserts in-order
    if all(isinstance(o, dict) and "_span_start" in o for o in (new_lines or [])):
        for o in new_lines:
            pos = _clamp(int(o.get("_span_start", len(dialogues))),
                         0, len(dialogues))
            if logger:
                logger(
                    f"[REINJ_LINE] pos={pos} speaker={o.get('character','?')} text={o.get('text','')[:60]}")
            dialogues[pos:pos] = [o]
        return dialogues

    # 2) Group-level positional guess
    approx_pos = None
    if isinstance(start_index, int):
        approx_pos = _clamp(start_index, 0, len(dialogues))
    elif isinstance(end_index, int):
        approx_pos = _clamp(end_index, 0, len(dialogues))
    else:
        approx_pos = len(dialogues)  # tail as last resort

    # Try to anchor the FIRST new line near approx_pos
    first_txt = _n((new_lines[0] or {}).get("text", "")) if new_lines else ""
    anchor_pos = None

    # 2a) Direct positional insert if we have no text to match
    best_i, best_sim = None, 0.0  # ensure defined for logging below

    if not first_txt:
        anchor_pos = approx_pos
    else:
        # 3) Fuzzy search in a local window around approx_pos
        window = 12
        for i in range(max(0, approx_pos - window), min(len(dialogues), approx_pos + window + 1)):
            cand = _n((dialogues[i] or {}).get("text", ""))
            if not cand:
                continue
            if first_txt in cand or cand in first_txt:
                best_i, best_sim = i, 1.0
                break
            sim = SequenceMatcher(None, cand, first_txt).ratio()
            if sim > best_sim:
                best_i, best_sim = i, sim

        if best_i is not None and best_sim >= 0.80:
            anchor_pos = best_i
        else:
            anchor_pos = approx_pos  # fall back to positional guess

    # Insert preserving order starting at anchor_pos
    try:
        _reason = (
            "tail" if anchor_pos == len(dialogues)
            else ("fuzzy" if (best_i is not None and best_sim >= 0.80) else "positional")
        )
        print(
            f"[REINJ_ANCHOR] approx_pos={approx_pos} anchor_pos={anchor_pos} reason={_reason}", flush=True)
    except Exception:
        pass

    if logger:
        logger(
            f"[REINJ_LINE] pos={anchor_pos} (positional-first) text='{(new_lines[0] or {}).get('text','')[:60]}'")
    if logger:
        logger(
            f"[REINJ_LINE] pos={anchor_pos} (positional-first) text='{(new_lines[0] or {}).get('text','')[:60]}'")

    # Clamp insertion index without affecting logs
    insert_pos = anchor_pos
    try:
        # prevent big forward jumps unless it's a clear containment match
        jump = insert_pos - approx_pos
        if jump > 3:
            _n = _norm_for_compare
            first_txt_cmp = _n((new_lines[0] or {}).get("text", "") or "")
            cand_txt_cmp = _n(
                (dialogues[insert_pos] or {}).get("text", "") or "")
            if not (first_txt_cmp and (first_txt_cmp in cand_txt_cmp or cand_txt_cmp in first_txt_cmp)):
                insert_pos = min(len(dialogues), approx_pos + 1)
    except Exception:
        pass

    try:
        if anchor_pos == len(dialogues):
            ft = _norm_for_compare((new_lines[0] or {}).get("text", "") or "")
            if ft:
                recent = dialogues[max(0, anchor_pos-50):anchor_pos]
                for ln in recent:
                    prev = _norm_for_compare((ln or {}).get("text", "") or "")
                    if prev and (ft in prev or prev in ft):
                        try:
                            print("[REINJ_SKIP] reason=tail_dup", flush=True)
                        except Exception:
                            pass
                        return dialogues
    except Exception:
        pass

    # Final guards per-candidate before appending: quote-covered and attribution-only
    guarded: list = []
    try:
        # Build parsed_norm_texts for quote coverage using existing dialogues as the parsed base
        parsed_norm_texts = [_norm_for_compare_punct_neutral(
            (ln or {}).get("text", "")) for ln in dialogues]
        for _cand in (new_lines or []):
            candidate_text = (_cand or {}).get("text", "") or ""
            # Skip if quoted core already covered by parsed lines
            qs = _extract_quotes(candidate_text)
            if qs:
                qs_norm = [_norm_for_compare_punct_neutral(q) for q in qs]
                if any(qn and any(qn == pt for pt in parsed_norm_texts) for qn in qs_norm):
                    try:
                        print(
                            "[REINJ_SKIP] reason=fb_quoted_already_covered", flush=True)
                    except Exception:
                        pass
                    continue
            # Skip attribution-only tails
            if _fb_is_attrib_only(candidate_text):
                try:
                    print("[REINJ_SKIP] reason=fb_attrib_tail", flush=True)
                except Exception:
                    pass
                continue
            guarded.append(_cand)
    except Exception:
        guarded = list(new_lines or [])

    if not guarded:
        return dialogues

    before = len(dialogues)
    try:
        dialogues[insert_pos:insert_pos] = guarded
    except Exception as e:
        import traceback
        try:
            print(f"[REINJ_FAIL][FALLBACK] {e}", flush=True)
        except Exception:
            pass
        try:
            traceback.print_exc()
        except Exception:
            pass
        # safety: still keep the lines rather than losing them
        try:
            dialogues.extend([{**ln, "_src": ln.get("_src", "fb")}
                             for ln in (new_lines or [])])
        except Exception:
            # last-resort: append as-is
            try:
                dialogues.extend(list(new_lines or []))
            except Exception:
                pass
    try:
        print(f"[REINJ_OK] added={len(dialogues)-before}", flush=True)
    except Exception:
        pass
    return dialogues
