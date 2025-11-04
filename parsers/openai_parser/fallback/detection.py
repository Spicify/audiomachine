import re
from ..diag import diag_enabled, diag_print
from .textnorm import _collapse_ws, _norm_for_compare_punct_neutral, _normalize_text
from .parsing import _extract_quotes
from .diag_ctx import _DIAG_CTX


_SENT_SPLIT_RX = re.compile(r'(?<=[.!?])\s+')


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
        # Initialize similarity to 1.0 (will be updated if not covered)
        si["sim"] = 1.0
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
            # Store similarity for anchoring correction
            if not covered:
                best_sim = 0.0
                for pt in parsed_norm_texts:
                    if not pt:
                        continue
                    best_sim = max(best_sim, _token_similarity(norm_sent, pt))
                si["sim"] = best_sim
            # else: sim already initialized to 1.0 at start
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
        # Adjust start index when entire chunk was incorrectly marked missing
        if seg.get("kind") == "missing" and start_idx == 0 and len(parsed_lines) > 5:
            try:
                # Find first unmatched sentence with lowest similarity from coverage map
                lowest_sim = 1.0
                best_idx = 0
                for i, si in enumerate(sent_infos):
                    sim = si.get("sim", 1.0)
                    if sim < 0.6 and sim < lowest_sim:
                        lowest_sim = sim
                        best_idx = i
                if best_idx > 0:
                    start_idx = best_idx
                    seg["start_idx"] = start_idx
                    if diag_enabled():
                        diag_print(f"[ANCHOR_FIX] corrected start_idx={start_idx} (prev=0, sim={lowest_sim:.2f})")
            except Exception:
                pass
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
