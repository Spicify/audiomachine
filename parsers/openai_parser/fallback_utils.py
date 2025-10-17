import os
from openai import OpenAI
import time
import re
from dotenv import load_dotenv

# Ensure .env is loaded before reading environment variables
load_dotenv()
print("[DEBUG] .env loaded for Frendli fallback")


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
        "Emotions: Exactly TWO per line; if none, use ['neutral','calm'].",
        f"Known characters: [{chars_str}]",
        "Examples:",
        '{"character":"Narrator","emotions":["neutral","calm"],"text":"The rain poured outside."}',
        '{"character":"Bella","emotions":["confused","tense"],"text":"I can’t believe he said that."}',
        '{"character":"Ambiguous","emotions":["neutral","calm"],"candidates":["Aria","Luca"],"text":"You two need to keep quiet."}',
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
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


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


def _norm_for_compare(s: str) -> str:
    s = _normalize_text(s)
    # strip surrounding quotes and collapse spaces are already done in _normalize_text
    return s.rstrip('.,;:!?')


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


def filter_fallback_lines(segment_text: str, candidate_lines: list[dict]) -> list[dict]:
    """Filter Friendli candidate lines against the source segment with robust fuzzy match.

    Acceptance:
    - edit ratio ≥ 0.55 OR
    - short utterances (<6 tokens): token jaccard ≥ 0.6
    - if both ≤4 tokens and share ≥1 token → accept
    Logs one [FB_FUZZY] per candidate.
    """
    src_quotes = _extract_quotes(segment_text)
    src = src_quotes[0] if src_quotes else segment_text
    src_n = _norm_for_compare(src)
    kept: list[dict] = []
    for ln in candidate_lines or []:
        txt = (ln or {}).get("text", "")
        cand_n = _norm_for_compare(txt)
        jacc = _token_jaccard(src_n, cand_n)
        edit = _edit_ratio(src_n, cand_n)
        cand_tokens = len((cand_n or "").split())
        keep = False
        if edit >= 0.55:
            keep = True
        elif cand_tokens < 6 and jacc >= 0.6:
            keep = True
        else:
            both_short = len((src_n or "").split()) <= 4 and cand_tokens <= 4
            shared = jacc > 0.0
            if both_short and shared:
                keep = True
        try:
            print(
                f"[FB_FUZZY] src='{src_n[:60]}' cand='{cand_n[:60]}' jacc={jacc:.2f} edit={edit:.2f} keep={keep}",
                flush=True,
            )
        except Exception:
            pass
        if keep:
            kept.append(ln)
    return kept


_SENT_SPLIT_RX = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> list[dict]:
    parts: list[dict] = []
    if not text:
        return parts
    # Normalize whitespace without altering content semantics
    norm_text = re.sub(r"\s+", " ", text.strip())
    chunks = re.split(_SENT_SPLIT_RX, norm_text)
    idx = 0
    for c in chunks:
        c = (c or "").strip()
        if not c:
            continue
        norm = _normalize_text(c)
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
    parsed_norm_texts = [_normalize_text(
        d.get("text", "")) for d in parsed_lines]

    uncovered_indices: list[int] = []
    for si in sent_infos:
        norm_sent = si["norm"]
        if not norm_sent:
            continue
        covered = any(
            norm_sent in pt or pt in norm_sent for pt in parsed_norm_texts if pt)
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

    def _n(s: str) -> str:
        if not s:
            return ""
        s = s.replace("\u201C", '"').replace("\u201D", '"')
        s = s.replace("\u2018", "'").replace("\u2019", "'")
        s = re.sub(r"\s+", " ", s).strip().lower()
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
    if not first_txt:
        anchor_pos = approx_pos
    else:
        # 3) Fuzzy search in a local window around approx_pos
        window = 8
        best_i, best_sim = None, 0.0
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
    if logger:
        logger(
            f"[REINJ_LINE] pos={anchor_pos} (positional-first) text='{(new_lines[0] or {}).get('text','')[:60]}'")
    dialogues[anchor_pos:anchor_pos] = new_lines
    return dialogues
