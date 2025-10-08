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


def replace_or_insert_lines(dialogues, start_index, new_lines, end_index=None):
    """
    Inserts or replaces fallback-parsed lines into dialogues list using semantic overlap and content anchoring.
    The start/end indices are treated as sentence anchors, not direct list positions.
    """
    try:
        span = f"[{start_index}:{(end_index if end_index is not None else start_index)+1}]"
        print(
            f"[DEBUG] Inserting {len(new_lines)} line(s) at span {span}", flush=True)
    except Exception:
        pass
    if end_index is None:
        end_index = start_index

    def _n(s: str) -> str:
        return _normalize_text(s or "")

    # Diagnostics: snapshot before
    try:
        _lo = max(0, start_index - 2)
        _hi = min(len(dialogues),
                  (end_index if end_index is not None else start_index) + 3)
        _before = [(i, str(dialogues[i].get("character", "")), (dialogues[i].get(
            "text", "") or "")[:80]) for i in range(_lo, _hi)]
        print(
            f"[DIAG] Before replace/insert slice {_lo}:{_hi}: {_before}", flush=True)
    except Exception:
        pass

    new_norms = [_n(nl.get("text", ""))
                 for nl in (new_lines or []) if (nl and nl.get("text"))]
    # Remove semantic overlaps and REJECTED anywhere in dialogues
    i = 0
    while i < len(dialogues):
        try:
            d = dialogues[i]
            if str(d.get("character", "")).upper() == "REJECTED":
                del dialogues[i]
                continue
            dnorm = _n(d.get("text", ""))
            if dnorm and any((dnorm == nn or dnorm in nn or nn in dnorm) for nn in new_norms):
                del dialogues[i]
                continue
        except Exception:
            pass
        i += 1

    # Determine insertion point by content anchoring using the first new line
    insert_at = None
    first_new = new_norms[0] if new_norms else ""
    if first_new:
        for j in range(len(dialogues) - 1, -1, -1):
            try:
                dnorm = _n(dialogues[j].get("text", ""))
                if dnorm and (dnorm in first_new or first_new in dnorm):
                    insert_at = j + 1
                    break
            except Exception:
                continue
    if insert_at is None:
        if start_index == 0:
            insert_at = 0
        else:
            insert_at = len(dialogues)
            try:
                print(
                    "[DIAG] No content anchor found; appending new lines at end.", flush=True)
            except Exception:
                pass

    dialogues[insert_at:insert_at] = list(new_lines or [])

    # Diagnostics: snapshot after
    try:
        _lo2 = max(0, insert_at - 2)
        _hi2 = min(len(dialogues), (insert_at + len(new_lines) + 3))
        _after = [(i, str(dialogues[i].get("character", "")), (dialogues[i].get(
            "text", "") or "")[:80]) for i in range(_lo2, _hi2)]
        print(
            f"[DIAG] After replace/insert slice {_lo2}:{_hi2}: {_after}", flush=True)
    except Exception:
        pass
