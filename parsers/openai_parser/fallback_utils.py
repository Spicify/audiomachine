import os
from openai import OpenAI
import time
import re
from dotenv import load_dotenv
from utils.text_normalizer import normalize_text as _norm_for_compare
from .diag import diag_enabled, diag_print, nsfw_marker_present, preview, approx_token_len

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
    if diag_enabled():
        try:
            _stop = None
            diag_print(
                f"[LLM_REQ] engine=Friendli chunk_idx={_DIAG_CTX.get('chunk_idx','-')} chunk_first50='{preview(user_prompt,50)}' prompt_len_chars={len(user_prompt or '')} prompt_len_tokens~={approx_token_len(user_prompt or '')} params={{model:'mistralai/Mistral-Small-3.1-24B-Instruct-2503'}} nsfw_in_prompt={nsfw_marker_present(user_prompt or '')}"
            )
        except Exception:
            pass
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
        if diag_enabled():
            try:
                http_status = getattr(
                    getattr(response, "_response", None), "status_code", None)
            except Exception:
                http_status = None
            diag_print(
                f"[LLM_RESP] engine=Friendli http_status={http_status or 'n/a'} elapsed_ms={elapsed*1000.0:.0f} content_len={len(out)} nsfw_in_resp={nsfw_marker_present(out)}"
            )
            if not out:
                diag_print(
                    f"[LLM_EMPTY] engine=Friendli chunk_idx={_DIAG_CTX.get('chunk_idx','-')}")
        print(
            f"[DEBUG] Frendli request end ({elapsed*1000.0:.0f} ms, chars={len(out)})", flush=True)
        return out
    except Exception as e:
        if diag_enabled():
            diag_print(
                f"[LLM_ERR] engine=Friendli err={type(e).__name__}:{str(e)[:200]}")
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


_DIAG_CTX: dict = {"chunk_idx": None, "tail_appends": {}}


def _set_diag_context(*, chunk_idx: int | None = None) -> None:
    if not diag_enabled():
        return
    try:
        if chunk_idx is not None:
            _DIAG_CTX["chunk_idx"] = int(chunk_idx)
    except Exception:
        pass


def _bump_tail_append() -> None:
    if not diag_enabled():
        return
    try:
        ci = _DIAG_CTX.get("chunk_idx")
        if ci is None:
            return
        _DIAG_CTX.setdefault("tail_appends", {})
        _DIAG_CTX["tail_appends"][ci] = 1 + \
            int(_DIAG_CTX["tail_appends"].get(ci, 0))
    except Exception:
        pass


def _get_tail_appends(chunk_idx: int) -> int:
    try:
        return int(_DIAG_CTX.get("tail_appends", {}).get(chunk_idx, 0))
    except Exception:
        return 0


def _split_sentences(text: str) -> list[dict]:
    parts: list[dict] = []
    if not text:
        return parts
    # Normalize whitespace without altering content semantics
    norm_text = _collapse_ws(text)
    chunks = re.split(_SENT_SPLIT_RX, norm_text)
    idx = 0
    import hashlib as _hashlib
    for c in chunks:
        c = (c or "").strip()
        if not c:
            continue
        norm = _norm_for_compare_punct_neutral(c)
        if norm and norm not in {'"', "'"}:
            sid_src = norm.lower().encode("utf-8", "ignore")
            sent_id = _hashlib.sha1(sid_src).hexdigest()[:10]
            parts.append({"index": idx, "text": c,
                         "norm": norm, "sent_id": sent_id})
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


def detect_missing_or_rejected_lines(chunk_text, parsed_lines, *, ledger: list | None = None, chapter_char_offset: int = 0, **kwargs):
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
        reason = "prefix" if covered else "none"

        # QUOTE-AWARE coverage: if the sentence has a quoted core that matches any parsed text, consider it covered
        if not covered:
            quoted_spans = _extract_quotes(si["text"])
            if quoted_spans:
                quoted_norms = [_norm_for_compare_punct_neutral(
                    q) for q in quoted_spans]
                if any(qn and any(qn == pt for pt in parsed_norm_texts) for qn in quoted_norms):
                    covered = True
                    reason = "verbatim"
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
                    reason = f"fuzzy"
                    break
            if not covered:
                try:
                    print(
                        f"[DIAG][SIM] missing segment detected by token similarity (no match ≥0.6): '{norm_sent[:80]}'", flush=True)
                except Exception:
                    pass
        # [COVERAGE] log per-sentence decision
        if diag_enabled():
            try:
                _ll = "unavailable"
                if isinstance(ledger, list):
                    # best-effort: find ledger idx by start_char/end_char containment against si text span if we had spans
                    # we don't have direct spans from here, so just note unavailable to avoid behavior change
                    _ll = "unavailable"
                diag_print(
                    f"[COVERAGE] sent_id={si.get('sent_id','??????????')} chunk_idx={_DIAG_CTX.get('chunk_idx','-')} covered={bool(covered)} reason={reason} sim={'-' if reason!='fuzzy' else f'{sim:.2f}' if 'sim' in locals() else '-'}")
                diag_print(f"[LEDGER_LINK] {_ll}")
            except Exception:
                pass
        if not covered:
            uncovered_indices.append(si["index"])

    missing_groups = _group_consecutive(uncovered_indices)
    missing_segments = []
    for start_idx, end_idx in missing_groups:
        joined_text = "\n".join(
            si["text"] for si in sent_infos if start_idx <= si["index"] <= end_idx)
        seg = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "text": joined_text,
            "kind": "missing",
        }
        missing_segments.append(seg)
        if diag_enabled():
            try:
                # choose the first sentence id in the span for reference
                sid = next((si.get("sent_id")
                           for si in sent_infos if si["index"] == start_idx), "??????????")
                diag_print(
                    f"[REINJECT_DECIDE] sent_id={sid} target_idx={start_idx} reason=missing")
            except Exception:
                pass

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
                seg = {
                    "start_idx": best_i,
                    "end_idx": best_i,
                    "text": rtxt,
                    "kind": "rejected",
                }
                rejected_segments.append(seg)
                if diag_enabled():
                    try:
                        diag_print(
                            f"[REINJECT_DECIDE] sent_id={'-' } target_idx={best_i} reason=rejected")
                    except Exception:
                        pass

    # Ledger window binding for segments (best-effort)
    if isinstance(ledger, list) and ledger:
        try:
            from parsers.openai_parser.utils_misc import normalize_for_sid as _norm_sid
        except Exception:
            def _norm_sid(s: str) -> str:
                import re as __re
                return __re.sub(r"\s+", " ", (s or "").strip().lower())

        _ledger_norms = [_norm_sid((d or {}).get("text", "")) for d in ledger]
        for _seg in (missing_segments + rejected_segments):
            try:
                a = int(_seg.get("start_idx", 0))
                b = int(_seg.get("end_idx", a))
            except Exception:
                a, b = 0, 0
            if b < a:
                b = a
            _cand_idxs: list[int] = []
            if a <= b and sent_infos:
                for si in sent_infos:
                    if not (a <= si["index"] <= b):
                        continue
                    sn = _norm_sid(si.get("text", ""))
                    try:
                        pos = _ledger_norms.index(sn)
                    except ValueError:
                        pos = -1
                    if pos >= 0:
                        _cand_idxs.append(pos)
            if not _cand_idxs:
                # Fallback: containment search from the segment combined text
                seg_text_norm = _norm_sid(_seg.get("text", ""))
                first_idx = None
                last_idx = None
                for i, ln in enumerate(_ledger_norms):
                    if ln and (ln in seg_text_norm or seg_text_norm in ln):
                        if first_idx is None:
                            first_idx = i
                        last_idx = i
                if first_idx is not None and last_idx is not None:
                    _cand_idxs = list(range(first_idx, last_idx+1))
            if _cand_idxs:
                first_idx = min(_cand_idxs)
                last_idx = max(_cand_idxs)
            else:
                first_idx = None
                last_idx = None
            _seg["ledger_start_idx"] = first_idx
            _seg["ledger_end_idx"] = last_idx
            if diag_enabled():
                try:
                    seg_id = f"{_seg.get('kind','seg')}:{a}-{b}"
                    size = (last_idx - first_idx +
                            1) if (first_idx is not None and last_idx is not None) else 0
                    diag_print(
                        f"[SEG_LEDGER_WINDOW] seg_id={seg_id} start_idx={first_idx} end_idx={last_idx} size={size}")
                except Exception:
                    pass

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


def _resolve_candidate_sid(candidate_text: str, ledger: list, start_idx: int, end_idx: int):
    """
    Return (sid, sid_span) where sid is a single ledger sid, or sid_span=(sid1, sid2) if the candidate merges adjacent sentences.
    Constrain search to [start_idx, end_idx]. Return (None, None) if nothing matches.
    """
    try:
        from parsers.openai_parser.utils_misc import normalize_for_sid as _norm_sid
    except Exception:
        def _norm_sid(s: str) -> str:
            import re as __re
            return __re.sub(r"\s+", " ", (s or "").strip().lower())

    cand_norm = _norm_sid(candidate_text or "")
    if not isinstance(ledger, list) or not ledger:
        return None, None
    lo = max(0, int(start_idx or 0))
    hi = min(len(ledger)-1, int(end_idx or 0)
             ) if end_idx is not None else len(ledger)-1

    # 1) direct containment of a single ledger sentence
    for i in range(lo, hi+1):
        try:
            sn = _norm_sid((ledger[i] or {}).get("text", ""))
            if sn and (sn in cand_norm or cand_norm in sn):
                return (ledger[i].get("sid"), None)
        except Exception:
            continue

    # 2) adjacent concatenation (two-sentence merge)
    glues = [" ", ", ", '" ', ' "']
    for i in range(lo, max(lo, hi)):
        try:
            a = _norm_sid((ledger[i] or {}).get("text", ""))
            b = _norm_sid((ledger[i+1] or {}).get("text", ""))
        except Exception:
            continue
        for g in glues:
            joined = f"{a}{g}{b}".strip()
            if joined and (joined in cand_norm or cand_norm in joined):
                return (None, (ledger[i].get("sid"), ledger[i+1].get("sid")))

    # 3) quote-like partial: attempt text containment only (no spans)
    t = candidate_text or ""
    looks_quote = (t.startswith('"') or t.startswith('\u201C')
                   or t.endswith('"') or t.endswith('\u201D') or ('"' in t))
    if looks_quote:
        for i in range(lo, hi+1):
            try:
                sn = _norm_sid((ledger[i] or {}).get("text", ""))
            except Exception:
                continue
            if sn and (sn in cand_norm or cand_norm in sn):
                return (ledger[i].get("sid"), None)
        return None, None

    return None, None


def annotate_candidates_with_sid(candidates: list, *, ledger: list | None, seg_obj: dict, chunk_idx: int | None = None) -> tuple[int, int]:
    """Annotate candidate dicts with _target_sid/_target_sid_span within the segment's ledger window.
    Logs per-candidate resolution and returns (mapped_count, unmapped_count).
    No behavior change; purely diagnostic tagging.
    """
    mapped, unmapped = 0, 0
    if not isinstance(candidates, list):
        return mapped, unmapped
    ls = seg_obj.get("ledger_start_idx") if isinstance(seg_obj, dict) else None
    le = seg_obj.get("ledger_end_idx") if isinstance(seg_obj, dict) else None
    seg_id = f"{seg_obj.get('kind','seg')}:{seg_obj.get('start_idx',0)}-{seg_obj.get('end_idx',0)}" if isinstance(
        seg_obj, dict) else "seg:?"
    for j, cand in enumerate(candidates):
        if ls is None or le is None or not (isinstance(ledger, list) and ledger):
            if diag_enabled():
                try:
                    diag_print(
                        f"[SID_RESOLVE_FAIL] seg={seg_id} cand_idx={j} reason=no-window window=({ls},{le})")
                except Exception:
                    pass
            cand["_target_sid"] = None
            cand["_target_sid_span"] = None
            unmapped += 1
            continue
        sid, sid_span = _resolve_candidate_sid(
            (cand or {}).get("text", ""), ledger, int(ls), int(le))
        if sid:
            cand["_target_sid"] = sid
            cand["_target_sid_span"] = None
            mapped += 1
            if diag_enabled():
                try:
                    diag_print(
                        f"[SID_RESOLVE] seg={seg_id} cand_idx={j} sid={sid} span={'-'} preview='{preview((cand or {}).get('text',''),80)}'")
                except Exception:
                    pass
        elif sid_span:
            cand["_target_sid"] = None
            cand["_target_sid_span"] = tuple(sid_span)
            mapped += 1
            if diag_enabled():
                try:
                    diag_print(
                        f"[SID_RESOLVE] seg={seg_id} cand_idx={j} sid={'-'} span={sid_span} preview='{preview((cand or {}).get('text',''),80)}'")
                except Exception:
                    pass
        else:
            cand["_target_sid"] = None
            cand["_target_sid_span"] = None
            unmapped += 1
            if diag_enabled():
                try:
                    diag_print(
                        f"[SID_RESOLVE_FAIL] seg={seg_id} cand_idx={j} reason=no-match window=({ls},{le})")
                except Exception:
                    pass
    return mapped, unmapped


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
            if diag_enabled():
                try:
                    diag_print(
                        f"[REINJECT_APPLY] sent_id={'-'} at={pos} mode=replace")
                except Exception:
                    pass
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
        if diag_enabled() and anchor_pos == len(dialogues):
            _bump_tail_append()
            diag_print(
                f"[TAIL_APPEND] sent_id={'-'} chunk_idx={_DIAG_CTX.get('chunk_idx','-')} why={'coverage_failed' if _reason=='tail' else _reason}")
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
        if diag_enabled():
            for k, _cand in enumerate(guarded):
                try:
                    diag_print(
                        f"[REINJECT_APPLY] sent_id={'-'} at={insert_pos + k} mode={'append_tail' if anchor_pos==len(dialogues) else 'insert'}")
                except Exception:
                    pass
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
