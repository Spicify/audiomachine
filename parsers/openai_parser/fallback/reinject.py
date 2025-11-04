import re
from difflib import SequenceMatcher
from ..diag import diag_enabled, diag_print
from utils.text_normalizer import normalize_text as _norm_for_compare
from .textnorm import _collapse_ws, _norm_for_compare_punct_neutral
from .parsing import _extract_quotes
from .diag_ctx import _bump_tail_append, _DIAG_CTX


_ATTRIB_ONLY_RE = re.compile(
    r"^(?:he|she|they|[A-Z][a-z]+)\s+(?:said|asked|hissed|murmured|replied|whispered|shouted|yelled|growled|muttered|breathed|ordered|instructed)\.?$",
    re.IGNORECASE,
)


def _is_attribution_only(text: str) -> bool:
    return bool(_ATTRIB_ONLY_RE.match((text or "").strip()))


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


def _direct_narration_reinject_ok(text: str) -> bool:
    t = (text or "").strip()
    if not t or ('"' in t) or ('\u201C' in t) or ('\u201D' in t):
        return False
    words = re.findall(r"[A-Za-z0-9’']+", t)
    return 1 <= len(words) <= 6


def replace_or_insert_lines(dialogues, new_lines, start_index=None, end_index=None, logger=None, *,
                            ledger: list | None = None,
                            seg_obj: dict | None = None,
                            seen_sids: set | None = None,
                            emitted_idx_by_sid: dict | None = None,
                            chunk_idx: int | None = None):
    """
    Insert or replace lines with priority:
    1) Per-line `_span_start` (deterministic).
    2) Group-level positional guess (start_index/end_index clamped).
    3) Fuzzy content anchor near the guessed position.
    4) Final fallback: tail-append.
    """
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

    # SID-anchored fast path before positional/fuzzy
    seen_sids = seen_sids if isinstance(seen_sids, set) else set()
    emitted_idx_by_sid = emitted_idx_by_sid if isinstance(
        emitted_idx_by_sid, dict) else {}
    try:
        sid_window = None
        if isinstance(seg_obj, dict):
            ls = seg_obj.get("ledger_start_idx")
            le = seg_obj.get("ledger_end_idx")
            if ls is not None and le is not None:
                sid_window = (int(ls), int(le))
    except Exception:
        sid_window = None

    sid_inserted = 0
    sid_dups_skipped = 0

    guarded_sid_first: list = []
    for _cand in list(new_lines or []):
        # skip attribution-only
        if _is_attribution_only((_cand or {}).get("text", "")):
            if diag_enabled():
                try:
                    diag_print(
                        f"[REINJ_SKIP_ATTRIB_ONLY] text='{((_cand or {}).get('text','') or '')[:80]}'")
                except Exception:
                    pass
            continue
        tgt_sid = (_cand or {}).get("_target_sid")
        tgt_span = (_cand or {}).get("_target_sid_span")
        if not tgt_sid and not tgt_span:
            guarded_sid_first.append(_cand)
            continue
        # dedup
        if tgt_sid and tgt_sid in seen_sids:
            sid_dups_skipped += 1
            if diag_enabled():
                try:
                    diag_print(f"[SID_DUP_SKIP] sid={tgt_sid}")
                except Exception:
                    pass
            continue
        anchor_sid = tgt_sid or (tgt_span[0] if isinstance(
            tgt_span, (list, tuple)) and len(tgt_span) == 2 else None)
        if sid_window is None and not emitted_idx_by_sid:
            # cannot anchor safely when no window and no context
            if diag_enabled():
                try:
                    diag_print(f"[SID_OUT_OF_RANGE] sid={anchor_sid}")
                except Exception:
                    pass
            continue
        # compute insertion index from emitted_idx_by_sid
        target_index = None
        if isinstance(ledger, list) and anchor_sid:
            try:
                # ledger index of target sid
                target_idx = None
                for i, d in enumerate(ledger):
                    if (d or {}).get("sid") == anchor_sid:
                        target_idx = i
                        break
                if target_idx is None:
                    if diag_enabled():
                        diag_print(f"[SID_OUT_OF_RANGE] sid={anchor_sid}")
                    continue
                # find nearest prev and next emitted sid indices
                prev_idx = max([idx for sid, idx in emitted_idx_by_sid.items()
                                if isinstance(idx, int) and any((e.get("sid") == sid for e in ledger if isinstance(e, dict))) and
                                next((i for i, d in enumerate(ledger) if (d or {}).get("sid") == sid), -1) < target_idx] or [-1])
                next_idx = min([idx for sid, idx in emitted_idx_by_sid.items()
                                if isinstance(idx, int) and any((e.get("sid") == sid for e in ledger if isinstance(e, dict))) and
                                next((i for i, d in enumerate(ledger) if (d or {}).get("sid") == sid), 1 << 30) > target_idx] or [len(dialogues)])
                if prev_idx >= 0:
                    target_index = min(len(dialogues), prev_idx + 1)
                    anchor_label = "prev"
                elif next_idx < len(dialogues):
                    target_index = max(0, next_idx)
                    anchor_label = "next"
                else:
                    target_index = 0
                    anchor_label = "start"
            except Exception:
                target_index = None
        if target_index is None:
            if diag_enabled():
                try:
                    diag_print(f"[SID_OUT_OF_RANGE] sid={anchor_sid}")
                except Exception:
                    pass
            continue
        try:
            dialogues[target_index:target_index] = [_cand]
            sid_inserted += 1
            if anchor_sid:
                seen_sids.add(anchor_sid)
                emitted_idx_by_sid[anchor_sid] = target_index
            if diag_enabled():
                try:
                    # include segment ledger window if available
                    ls = seg_obj.get("ledger_start_idx") if isinstance(
                        seg_obj, dict) else None
                    le = seg_obj.get("ledger_end_idx") if isinstance(
                        seg_obj, dict) else None
                    window_str = f"({ls},{le})"
                    if tgt_sid:
                        diag_print(
                            f"[SID_INSERT] sid={tgt_sid} at={target_index} anchor={anchor_label} window={window_str}")
                    elif tgt_span:
                        diag_print(
                            f"[SID_INSERT] span={tgt_span} at={target_index} anchor={anchor_label} window={window_str}")
                except Exception:
                    pass
        except Exception:
            continue

    # keep only those not handled by sid path for the rest of the logic
    new_lines = guarded_sid_first

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
            _nfc = _norm_for_compare
            first_txt_cmp = _nfc((new_lines[0] or {}).get("text", "") or "")
            cand_txt_cmp = _nfc(
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
            # Attribution-only guard
            if _is_attribution_only(candidate_text):
                if diag_enabled():
                    try:
                        diag_print(
                            f"[REINJ_SKIP_ATTRIB_ONLY] text='{(candidate_text or '')[:80]}'")
                    except Exception:
                        pass
                continue
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
        # If no candidates survive and we have an easy direct-narration sentence from the segment, inject by sid
        try:
            if isinstance(seg_obj, dict) and isinstance(ledger, list):
                ls = seg_obj.get("ledger_start_idx")
                le = seg_obj.get("ledger_end_idx")
                if ls is not None and le is not None and ls == le:
                    seg_sentence_text = (seg_obj.get("text", "") or "").strip()
                    # If the segment text comprises multiple sentences, pick the ledger one
                    seg_sentence_text = (ledger[int(ls)] or {}).get(
                        "text", "") or seg_sentence_text
                    if _direct_narration_reinject_ok(seg_sentence_text):
                        sid = (ledger[int(ls)] or {}).get("sid")
                        if sid and sid not in seen_sids:
                            # insert at sid anchor
                            anchor_label = "start"
                            target_index = 0
                            try:
                                # compute position same as above using emitted_idx_by_sid
                                target_idx = int(ls)
                                prev_idx = max([idx for s, idx in emitted_idx_by_sid.items()
                                                if isinstance(idx, int) and any((e.get("sid") == s for e in ledger if isinstance(e, dict))) and
                                                next((i for i, d in enumerate(ledger) if (d or {}).get("sid") == s), -1) < target_idx] or [-1])
                                next_idx = min([idx for s, idx in emitted_idx_by_sid.items()
                                                if isinstance(idx, int) and any((e.get("sid") == s for e in ledger if isinstance(e, dict))) and
                                                next((i for i, d in enumerate(ledger) if (d or {}).get("sid") == s), 1 << 30) > target_idx] or [len(dialogues)])
                                if prev_idx >= 0:
                                    target_index = min(
                                        len(dialogues), prev_idx + 1)
                                    anchor_label = "prev"
                                elif next_idx < len(dialogues):
                                    target_index = max(0, next_idx)
                                    anchor_label = "next"
                            except Exception:
                                pass
                            line = {"character": "Narrator", "emotions": [
                                "neutral", "neutral"], "text": seg_sentence_text}
                            dialogues[target_index:target_index] = [line]
                            seen_sids.add(sid)
                            emitted_idx_by_sid[sid] = target_index
                            if diag_enabled():
                                try:
                                    diag_print(
                                        f"[SID_DIRECT_NARR] sid={sid} at={target_index} anchor={anchor_label}")
                                except Exception:
                                    pass
                        else:
                            if diag_enabled():
                                try:
                                    diag_print(f"[SID_DUP_SKIP] sid={sid}")
                                except Exception:
                                    pass
            return dialogues
        except Exception:
            return dialogues

    before = len(dialogues)
    try:
        # Prefer in-place insert when we have a valid anchor; append only when no valid anchor
        if insert_pos is not None and 0 <= insert_pos <= len(dialogues):
            dialogues[insert_pos:insert_pos] = guarded
            if diag_enabled():
                try:
                    diag_print(f"[REINJECT_ORDER] inserted at {insert_pos} len={len(guarded)}")
                except Exception:
                    pass
            if diag_enabled():
                for k, _cand in enumerate(guarded):
                    try:
                        diag_print(
                            f"[REINJECT_APPLY] sent_id={'-'} at={insert_pos + k} mode={'append_tail' if anchor_pos==len(dialogues) else 'insert'}")
                    except Exception:
                        pass
        else:
            dialogues.extend(guarded)
            if diag_enabled():
                try:
                    diag_print(f"[TAIL_APPEND] applied len={len(guarded)} (no valid anchor)")
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
