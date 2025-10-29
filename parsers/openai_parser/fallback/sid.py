from ..diag import diag_enabled, diag_print, preview


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
