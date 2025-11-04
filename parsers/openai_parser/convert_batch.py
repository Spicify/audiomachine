from __future__ import annotations
from utils.text_normalizer import normalize_text as _norm_for_compare

from pathlib import Path
from typing import Any, Dict, List, Tuple
import datetime
import hashlib
import sys
import re as _re

from .core_types import RawParseResult, ParserState, DualLogger, _hash_key, _token_similarity, _attr_counts
from .prompt_builder import build_system_prompt, build_user_prompt
from .chunker import build_chunks, deduplicate_lines, deduplicate_lines_exact, diag_consume_dedup_conflicts
from .fallback import (
    detect_missing_or_rejected_lines,
    call_frendli_fallback,
    replace_or_insert_lines,
    filter_fallback_lines,
)
from .fallback.detection import _split_sentences as _fb_split_sents
from .fallback.textnorm import _norm_for_compare_punct_neutral as _fb_norm_cmp
from .fallback.parsing import _sanitize_character
from .openai_client import call_openai_safe, _save_debug_output
from .validator import validate_and_fix
from .utils_misc import _dedupe_chunk_boundaries, _build_sentence_to_pos_map, _simple_reinject_missing_as_narrator, _preclean_jsonl
from .utils_misc import build_sentence_ledger
from .chunker import _split_into_sentences_with_spans
from utils.session_logger import log_to_session, log_exception
from .diag import diag_enabled, diag_print, nsfw_marker_present, preview, approx_token_len
from .emotion_utils import ensure_two_emotions
from pathlib import Path as _P
import json as _json
import re as _re


def _promote_primary_emotion_from_context(chunk_text: str, line: dict) -> None:
    """Promote verb-mapped emotion to primary when an attribution verb is near the quote."""
    try:
        txt = (line or {}).get("text", "") or ""
        if not txt:
            return
        # normalize quotes
        hay = (chunk_text or "")
        hay = hay.replace("“", '"').replace(
            "”", '"').replace("‘", "'").replace("’", "'")
        needle = txt.replace("“", '"').replace(
            "”", '"').replace("‘", "'").replace("’", "'")
        needle = needle.strip().rstrip('.,;:!?"\'')
        i = hay.find(needle)
        if i < 0:
            return
        span = hay[max(0, i-80): i+len(needle)+80]

        # load verb->emotion mapping
        vpath = None
        for p in ("configs/verb_to_emotion.json", "../configs/verb_to_emotion.json", "../../configs/verb_to_emotion.json"):
            if _P(p).exists():
                vpath = p
                break
        if not vpath:
            return
        with open(vpath, "r", encoding="utf-8") as f:
            v2e = _json.load(f)
        verbs = "|".join(sorted([_re.escape(v)
                         for v in v2e.keys()], key=len, reverse=True))
        m = _re.search(rf"\b({verbs})\b", span, flags=_re.IGNORECASE)
        if not m:
            return
        emo = str(v2e.get(m.group(1).lower(), "")).strip()
        if not emo:
            return

        ems = list((line.get("emotions") or [])[:2])
        out = [emo]
        for e in ems:
            if e and e.lower() != emo.lower():
                out.append(e)
            if len(out) >= 2:
                break
        line["emotions"] = out[:2]
        try:
            print(
                f"[EMO_PROMOTE] verb='{m.group(1)}' → primary='{emo}' text='{needle[:60]}'", flush=True)
        except Exception:
            pass
    except Exception:
        return


def convert_batch(parser, raw_text: str) -> RawParseResult:
    # --- Start unified debug capture ---
    log_dir = Path("logs")
    try:
        log_dir.mkdir(exist_ok=True)
    except Exception:
        pass
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    log_path = log_dir / f"parser_run_{ts}.txt"

    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    _dual_logger = DualLogger(sys.stdout)
    sys.stdout = _dual_logger
    sys.stderr = _dual_logger
    print(f"[LOG_START] Parser debug capture started ({ts})", flush=True)

    try:
        log_to_session(
            "INFO", f"Parser convert start (chars={len(raw_text or '')})", src="parsers/openai_parser.py:convert")
    except Exception:
        pass
    warnings: List[str] = []
    errors: List[str] = []
    _diag_fb_calls: List[Dict[str, Any]] = []
    _diag_reinj_considered_total: int = 0
    _diag_reinj_added_total: int = 0
    if not raw_text or not raw_text.strip():
        return RawParseResult(formatted_text="", dialogues=[], stats={}, ambiguities=[], warnings=warnings, errors=errors)
    # Build canonical sentence ledger (optional; no behavior change)
    ledger = None
    try:
        ledger = build_sentence_ledger(
            raw_text, splitter=_split_into_sentences_with_spans)
    except Exception as _le:
        if diag_enabled():
            diag_print(f"[LEDGER_BUILD_ERR] {type(_le).__name__}: {_le}")

    chunks = build_chunks(
        raw_text, max_tokens=parser.max_tokens_per_chunk, model=parser.model, overlap_sentences=2)
    state = ParserState(known_characters=set(
        parser.default_known_characters))
    all_dialogues: List[Dict[str, Any]] = []
    per_chunk_dialogues: List[List[Dict[str, Any]]] = []
    ambiguities: List[Dict[str, Any]] = []

    def _state_summary(state_obj: ParserState) -> Dict[str, Any]:
        return {
            "recent_characters": list(state_obj.known_characters)[:20],
            "last_speaker": state_obj.last_speaker,
            "last_emotions": {k: v[-2:] for k, v in state_obj.last_emotions.items()},
            "unresolved": [a.get("text", "") for a in state_obj.unresolved_ambiguities][-5:],
        }

    def _parse_jsonl(raw_output: str) -> List[Dict[str, Any]]:
        lines = [ln.strip() for ln in raw_output.split("\n") if ln.strip()]
        objs: List[Dict[str, Any]] = []
        for ln in lines:
            # Skip AI refusal or moderation disclaimer lines (early)
            if isinstance(ln, str):
                lower_line = ln.lower()
                if any(
                    phrase in lower_line
                    for phrase in [
                        "i'm sorry",
                        "i can't assist",
                        "i cannot assist",
                        "as an ai",
                        "violates policy",
                        "explicit sexual content",
                        "involving familial",
                        "provide a non-sexual excerpt",
                    ]
                ):
                    if diag_enabled():
                        from .diag import diag_print
                        diag_print(f"[EARLY_SKIP_REFUSAL] text='{lower_line[:80]}...'")
                    continue
            if not (ln.startswith("{") and ln.endswith("}")):
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    # Also check parsed text field for refusal phrases
                    text_field = str((obj.get("text") or "")).lower()
                    if any(
                        phrase in text_field
                        for phrase in [
                            "i'm sorry",
                            "i can't assist",
                            "i cannot assist",
                            "as an ai",
                            "violates policy",
                            "explicit sexual content",
                            "involving familial",
                            "provide a non-sexual excerpt",
                        ]
                    ):
                        if diag_enabled():
                            from .diag import diag_print
                            diag_print(f"[EARLY_SKIP_REFUSAL] text='{text_field[:80]}...'")
                        continue
                    objs.append(obj)
            except Exception:
                continue
        if not objs:
            try:
                print(
                    "[WARN] _parse_jsonl: no valid JSONL lines extracted", flush=True)
            except Exception:
                pass
        return objs

    import json  # local to avoid altering module imports

    # For duplicate/relocation diagnostics across chapter-run
    _seen_sent_ids: dict[str, tuple[int, int]] = {}
    # SID state across chapter
    seen_sids: set[str] = set()
    emitted_idx_by_sid: dict[str, int] = {}

    # Running character cursor for chapter offsets (best-effort)
    # Prefer absolute offsets if available on Chunk; else use prefix sum of actual raw_text slices.
    char_cursor = 0
    for idx, ch in enumerate(chunks):
        # Set diag context for this chunk
        try:
            from .fallback.diag_ctx import _set_diag_context as _fb_set_ctx
            _fb_set_ctx(chunk_idx=idx)
        except Exception:
            pass
        if idx == len(chunks) - 1:
            print(
                "\n===== [EOD][CHUNK_TEXT] LAST CHUNK INPUT BEGIN =====", flush=True)
            try:
                print(ch.text, flush=True)
            except Exception:
                print("[EOD] <failed to print chunk text>", flush=True)
            print(
                "===== [EOD][CHUNK_TEXT] LAST CHUNK INPUT END =====\n", flush=True)
        summary = _state_summary(state)
        # Determine chunk start absolute char offset
        try:
            # Prefer a direct attribute if provided by chunker
            if hasattr(ch, "start_char"):
                chunk_start_char = int(getattr(ch, "start_char"))
            elif hasattr(ch, "abs_offset"):
                chunk_start_char = int(getattr(ch, "abs_offset"))
            else:
                chunk_start_char = int(char_cursor)
        except Exception:
            chunk_start_char = int(char_cursor)
        if diag_enabled():
            try:
                diag_print(
                    f"[CH_OFFSET] chunk_idx={idx} start_char={chunk_start_char}")
            except Exception:
                pass
        system_prompt = build_system_prompt(parser.allowed_emotions, list(
            state.known_characters), parser.include_narration, summary)
        user_prompt = build_user_prompt(ch.text, None)

        cache_key = _hash_key(ch.text, summary)
        if cache_key in parser._cache:
            raw_output = parser._cache[cache_key]
        else:
            try:
                raw_output = call_openai_safe(
                    system_prompt, user_prompt, client=parser.client, model=parser.model)
            except Exception as e:
                errors.append(
                    f"Parser API error on chunk {idx+1}/{len(chunks)}: {e}")
                try:
                    log_exception("parsers/openai_parser.py:convert", e)
                except Exception:
                    pass
                raise
            if parser.debug_save:
                _save_debug_output(
                    raw_output, suffix=f"_chunk{idx+1}", debug_save=parser.debug_save)
            parser._cache[cache_key] = raw_output

        if idx == len(chunks) - 1:
            print(
                "\n===== [EOD][OPENAI_RAW] LAST CHUNK RAW BEGIN (first 60 lines) =====", flush=True)
            try:
                for i, ln in enumerate((raw_output or "").splitlines()[:60], start=1):
                    print(f"{i:02d}: {ln}", flush=True)
            except Exception:
                print("[EOD] <failed to print raw output>", flush=True)
            print(
                "===== [EOD][OPENAI_RAW] LAST CHUNK RAW END =====\n", flush=True)
        items = _parse_jsonl(raw_output)
        if not items:
            raw_output = call_openai_safe(
                system_prompt + "\nIMPORTANT: Output JSONL ONLY.", user_prompt, client=parser.client, model=parser.model)
            if parser.debug_save:
                _save_debug_output(
                    raw_output, suffix=f"_retry_chunk{idx+1}", debug_save=parser.debug_save)
            items = _parse_jsonl(raw_output)

        try:
            from collections import Counter
            import os
            import json
            char_counts = Counter(
                [str((it or {}).get("character", "")).strip() or "<empty>" for it in (items or [])])
            _dbg_items_total = len(items or [])
            _dbg_rej = sum(1 for it in (items or []) if str(
                (it or {}).get("character", "")).strip().upper() == "REJECTED")
            print(
                f"[RAW_DIST] total={_dbg_items_total} counts={dict(char_counts)}", flush=True)

            # TEMP: extra log for first chunk
            try:
                _dbg_chunk_idx = (chunk_index if 'chunk_index' in locals()
                                  else (i if 'i' in locals() else 0))
                if _dbg_chunk_idx in (0, 1):  # depending on 0- or 1-based
                    for k, it in enumerate(items[:50]):
                        print(f"[RAW_FIRST50][{k:02d}] {it}", flush=True)
            except Exception:
                pass

            if _dbg_items_total > 0 and _dbg_rej / max(1, _dbg_items_total) >= 0.5:
                ts = time.strftime("%Y%m%d-%H%M%S")
                outdir = os.path.join("logs", "raw_chunks")
                os.makedirs(outdir, exist_ok=True)
                # chunk index var names differ; try both
                _dbg_chunk_idx = (chunk_index if 'chunk_index' in locals()
                                  else (i if 'i' in locals() else 0))
                path = os.path.join(
                    outdir, f"raw_chunk_{_dbg_chunk_idx}_{ts}.jsonl")
                with open(path, "w", encoding="utf-8") as f:
                    for it in (items or []):
                        try:
                            f.write(json.dumps(it, ensure_ascii=False) + "\n")
                        except Exception:
                            f.write(str(it) + "\n")
                print(f"[RAW_DUMP] path={path}", flush=True)
        except Exception as e:
            try:
                print(f"[RAW_DIST_ERR] {e}", flush=True)
            except Exception:
                pass

        for it in items:
            if str(it.get("character", "")).strip().lower() == "ambiguous":
                txt = (it.get("text") or "").strip()
                cands = [str(c).strip() for c in (
                    it.get("candidates") or []) if str(c).strip()]
                if not cands:
                    cands = ["Unknown"]
                ambiguities.append({
                    "id": f"amb-{idx+1}-{abs(hash(txt))}",
                    "text": txt,
                    "candidates": cands[:5],
                })

        fixed, warnings = validate_and_fix(
            items, warnings, state, kb=parser.kb, allowed_emotions=parser.allowed_emotions, memory=parser.memory)
        # Promote primary emotion from nearby speech verb context
        try:
            for _ln in fixed:
                chunk_txt = raw_text  # batch uses the current raw_text here
                if (_ln.get("character", " ").strip().lower() not in ("narrator", "rejected")) and (_ln.get("text") or ""):
                    _promote_primary_emotion_from_context(chunk_txt, _ln)
        except Exception:
            pass
        try:
            print(
                f"[ATTR_SUMMARY] phase=pre_fallback { _attr_counts(fixed) }", flush=True)
        except Exception:
            pass
        for _ln in fixed:
            try:
                _ln.setdefault("_src", "ai")
            except Exception:
                pass
        # Tag emitted OpenAI lines with sid where possible using local ledger window
        try:
            if ledger and isinstance(ledger, list):
                # compute quick window around this chunk using detection coverage or fallback: scan containment
                from .utils_misc import normalize_for_sid as _norm_sid
                lnorms = [_norm_sid((d or {}).get("text", "")) for d in ledger]
                for di, _ln in enumerate(fixed):
                    t = _norm_sid((_ln or {}).get("text", ""))
                    if not t:
                        continue
                    try:
                        si = -1
                        for i, ln in enumerate(lnorms):
                            if ln and (ln in t or t in ln):
                                si = i
                                break
                        if si >= 0:
                            sid = (ledger[si] or {}).get("sid")
                            if sid:
                                _ln.setdefault("_sid", sid)
                                if sid not in seen_sids:
                                    seen_sids.add(sid)
                                    emitted_idx_by_sid[sid] = len(
                                        per_chunk_dialogues[-1]) + di if per_chunk_dialogues else di
                    except Exception:
                        continue
        except Exception:
            pass

        try:
            # Pass ledger optionally (internal helper accepts extra arg; behavior unchanged if ignored)
            try:
                problem_segments = detect_missing_or_rejected_lines(
                    ch.text, fixed, ledger=ledger, chapter_char_offset=chunk_start_char)
            except TypeError:
                # older signature without ledger
                problem_segments = detect_missing_or_rejected_lines(
                    ch.text, fixed)
            print(
                f"[DEBUG] Fallback detection: {len(problem_segments)} segment(s) found", flush=True)
            if idx == len(chunks) - 1:
                try:
                    print(
                        f"[EOD][FB] segments_detected={len(problem_segments)} (last chunk)", flush=True)
                except Exception:
                    pass
            is_last_chunk = (idx == len(chunks) - 1)
            if is_last_chunk and problem_segments:
                filtered_segments = []
                sim_threshold = 0.5 if is_last_chunk else 0.6
                for seg in problem_segments:
                    seg_txt = (seg.get("text", "") or "").strip()
                    sim = 0.0
                    if seg_txt:
                        try:
                            sim = max(
                                (_token_similarity(seg_txt, s)
                                 for s in ch.text.splitlines() if s.strip()),
                                default=0.0,
                            )
                        except Exception:
                            sim = 0.0
                    if sim >= sim_threshold or len(seg_txt.split()) >= 4:
                        filtered_segments.append(seg)
                    else:
                        print(
                            f"[DIAG][FB_FILTER] Skipping weak EOD segment '{seg_txt}' (sim={sim:.2f})", flush=True)
                problem_segments = filtered_segments
                if len(problem_segments) > 3:
                    print(
                        f"[WARN][FB] Excessive fallback segments ({len(problem_segments)}) — merging into a single consolidated fallback request.", flush=True)
                    from .fallback.detection import _split_sentences as _fb_split
                    combined_text = " ".join(
                        seg.get("text", "") for seg in problem_segments[:15])
                    approx_indices: list[int] = []
                    for _seg in problem_segments[:15]:
                        try:
                            s_i = int(_seg.get("start_idx", 0))
                            e_i = int(_seg.get("end_idx", s_i))
                        except Exception:
                            s_i = 0
                            e_i = 0
                        if e_i < s_i:
                            e_i = s_i
                        span = list(range(s_i, e_i + 1)) or [s_i]
                        approx_indices.extend(span)
                    if not approx_indices:
                        try:
                            sent_infos = _fb_split(combined_text)
                            approx_indices = [0] * max(1, len(sent_infos))
                        except Exception:
                            approx_indices = [0]
                    try:
                        print(
                            f"[EOD_FALLBACK_SPANS] count={len(approx_indices)} indices={approx_indices[:10]}",
                            flush=True,
                        )
                    except Exception:
                        pass
                    problem_segments = [
                        {"text": combined_text, "_approx_indices": approx_indices}]
                    parser._eod_fallback_throttled = True
                else:
                    parser._eod_fallback_throttled = False
            if not problem_segments and any(
                d.get("character") == "REJECTED" for d in fixed
            ):
                print(
                    "[DIAG] WARNING: REJECTED present but detection found 0 segments.", flush=True)
        except Exception as _fe:
            try:
                log_exception("parsers/openai_parser.py:convert", _fe)
            except Exception:
                pass
            problem_segments = []
        if problem_segments:
            try:
                _ordered_segments = sorted(
                    problem_segments, key=lambda s: int(s.get("start_idx", 0)))
            except Exception:
                _ordered_segments = list(problem_segments)
            _running_delta = 0
            for _seg_i, seg in enumerate(_ordered_segments, start=1):
                try:
                    if idx == len(chunks) - 1:
                        print(
                            "\n----- [EOD][FB] Segment DETECTED (last chunk) -----", flush=True)
                        print(
                            f"[EOD][FB] seg#{_seg_i} kind={seg.get('kind','missing')} start_idx={seg.get('start_idx')} end_idx={seg.get('end_idx')}", flush=True)
                        print(
                            f"[EOD][FB] segment_text (first 300): {(seg.get('text','') or '')[:300]}", flush=True)
                    try:
                        print(
                            f"[DIAG] Known chars before fallback: {sorted(list(state.known_characters))[:20]}", flush=True)
                    except Exception:
                        pass
                    print(
                        f"[DIAG] Last speaker: {state.last_speaker}", flush=True)
                    _seg_text = seg.get("text", "") or ""
                    print(
                        f"[DIAG] Fallback segment text (len={len(_seg_text)}): {_seg_text[:300]}", flush=True)
                    print(
                        f"[DIAG] Passing known_characters to fallback: True", flush=True)
                    try:
                        _local_speakers = []
                        for _obj in (fixed or []):
                            _ch = str((_obj or {}).get(
                                "character", "")).strip()
                            if _ch and _ch.lower() not in ("narrator", "ambiguous", "rejected"):
                                _local_speakers.append(_ch)
                        scoped_known = sorted(set(_local_speakers))
                        if "Narrator" not in scoped_known:
                            scoped_known.append("Narrator")
                    except Exception:
                        scoped_known = ["Narrator"]
                    fb_raw = call_frendli_fallback(
                        system_prompt, seg.get("text", ""), known_characters=scoped_known)
                    print(
                        f"[DEBUG] Fallback raw returned length={len(fb_raw)}", flush=True)
                    if idx == len(chunks) - 1:
                        print(
                            "----- [EOD][FB] RAW Friendli OUT (first 30 lines) -----", flush=True)
                        try:
                            for j, ln in enumerate((fb_raw or '').splitlines()[:30], start=1):
                                print(f"{j:02d}: {ln}", flush=True)
                        except Exception:
                            print(
                                "[EOD] <failed to print friendli raw>", flush=True)
                    try:
                        _fb_lines_preview = (fb_raw or "").splitlines()[:2]
                        print(
                            f"[DIAG] Fallback RAW (first 2 lines): {_fb_lines_preview}", flush=True)
                    except Exception:
                        pass
                    if parser.debug_save:
                        try:
                            Path("debug_outputs").mkdir(exist_ok=True)
                            ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                            fname = f"fallback_raw_{ts}_chunk{idx+1}_seg{_seg_i}.txt"
                            fpath = Path("debug_outputs") / fname
                            with open(fpath, "w", encoding="utf-8") as _f:
                                _f.write(fb_raw or "")
                            print(
                                f"[DIAG] Saved fallback RAW to {fpath}", flush=True)
                        except Exception:
                            pass
                    # Tolerant Friendli parsing
                    try:
                        from .fallback.parsing import _parse_friendli_output as _fr_parse
                        fb_lines = _fr_parse(fb_raw)
                    except Exception:
                        fb_lines = _parse_jsonl(_preclean_jsonl(fb_raw))
                    try:
                        fb_lines = [_sanitize_character(
                            x, sorted(list(state.known_characters))) for x in fb_lines]
                    except Exception:
                        pass
                    print(
                        f"[DEBUG] Fallback parsed {len(fb_lines)} line(s)", flush=True)
                    fb_valid, warnings = validate_and_fix(
                        fb_lines, warnings, state, kb=parser.kb, allowed_emotions=parser.allowed_emotions, memory=parser.memory)
                    # Resolve SIDs for Friendli candidates using segment's ledger window if present
                    mapped_count = 0
                    unmapped_count = 0
                    try:
                        from .fallback.sid import annotate_candidates_with_sid as _annot_sid
                        if isinstance(fb_valid, list):
                            mapped_count, unmapped_count = _annot_sid(
                                fb_valid, ledger=ledger, seg_obj=seg, chunk_idx=idx)
                    except Exception:
                        mapped_count = mapped_count
                        unmapped_count = unmapped_count
                    if diag_enabled():
                        try:
                            diag_print(
                                f"[SID_MAP_SUMMARY] chunk_idx={idx} mapped={mapped_count} unmapped={unmapped_count}")
                        except Exception:
                            pass
                    try:
                        approx = seg.get("_approx_indices") or []
                        if approx and fb_valid:
                            for k, _ln in enumerate(fb_valid):
                                _ln["_span_start"] = int(
                                    approx[min(k, len(approx)-1)])
                    except Exception:
                        pass
                    segment_text = seg.get("text", "") or ""
                    from .fallback import filter_fallback_lines as _fb_filter
                    fb_valid = _fb_filter(segment_text, fb_valid)
                    print(
                        f"[EOD][FB_SUMMARY] After fuzzy filter: kept={len(fb_valid)}", flush=True)
                    for _ln in fb_valid:
                        try:
                            _ln.setdefault("_src", "fb")
                        except Exception:
                            pass
                    try:
                        sent_to_pos = _build_sentence_to_pos_map(
                            ch.text, fixed)
                        try:
                            from .fallback.detection import _split_sentences as _dbg_split
                            sents = _dbg_split(ch.text)
                            mapped = len(sent_to_pos)
                            total = len(sents)
                            pct = (mapped/total*100.0) if total else 0.0
                            unmapped = [i for i in range(
                                total) if i not in sent_to_pos][:5]
                            print(
                                f"[MAP_COVERAGE] sent_count={total} mapped={mapped} pct={pct:.1f}%", flush=True)
                            if unmapped:
                                print(
                                    f"[MAP_UNMAPPED] idxes={unmapped}", flush=True)
                        except Exception:
                            pass
                        anchor_sent = int(seg.get("start_idx", 0))
                        pos = sent_to_pos.get(anchor_sent)
                        if pos is None:
                            for d in range(1, 11):
                                if sent_to_pos.get(anchor_sent - d) is not None:
                                    pos = sent_to_pos.get(anchor_sent - d)
                                    break
                                if sent_to_pos.get(anchor_sent + d) is not None:
                                    pos = sent_to_pos.get(anchor_sent + d)
                                    break
                        if pos is None:
                            pos = min(max(anchor_sent, 0), len(fixed))
                        for _ln in fb_valid:
                            if isinstance(_ln, dict) and "_span_start" not in _ln:
                                _ln["_span_start"] = int(pos)
                    except Exception:
                        pass
                    try:
                        print(
                            f"[REINJ_CALL] fixed_len={len(fixed)} start_idx={seg.get('start_idx',0)} fb_valid_len={len(fb_valid)} end_idx={seg.get('end_idx', seg.get('start_idx',0))}", flush=True)
                    except Exception:
                        pass
                    if idx == len(chunks) - 1:
                        print(
                            f"[EOD][FB] validated_and_kept={len(fb_valid)} line(s) for seg#{_seg_i}", flush=True)
                    print(
                        f"[DEBUG] Fallback validated {len(fb_valid)} line(s)", flush=True)
                    try:
                        allowed = set(state.known_characters) | {
                            "Narrator", "Ambiguous"}
                        _unknown_count = 0
                        for _ln in fb_valid:
                            ch_name = str(_ln.get("character", ""))
                            if ch_name and ch_name not in allowed:
                                _txt = (_ln.get("text", "") or "")[:80]
                                print(
                                    f"[DIAG][UNKNOWN_SPEAKER] '{ch_name}' text='{_txt}' (allowed_size={len(allowed)})", flush=True)
                                _unknown_count += 1
                        if idx == len(chunks) - 1:
                            try:
                                print(
                                    f"[EOD][FB] unknown_speakers={_unknown_count}", flush=True)
                            except Exception:
                                pass
                    except Exception:
                        _unknown_count = 0
                    _seg_len = len(_seg_text)
                    _diag_fb_calls.append({
                        "chunk": idx + 1,
                        "seg": _seg_i,
                        "text_len": _seg_len,
                        "parsed": len(fb_valid),
                        "unknown": _unknown_count,
                    })
                    try:
                        print("[ATTR_SUMMARY] phase=post_fallback",
                              _attr_counts(fixed), flush=True)
                    except Exception:
                        pass
                    try:
                        logger = None
                    except Exception:
                        logger = None
                    try:
                        _base_pos = int(seg.get("start_idx", 0))
                    except Exception:
                        _base_pos = 0
                    _approx_pos = _base_pos
                    try:
                        print(
                            f"[REINJ_DELTA] base_pos={_base_pos} approx_pos={_approx_pos} delta={0}", flush=True)
                    except Exception:
                        pass
                    if fb_valid:
                        replace_or_insert_lines(
                            dialogues=fixed,
                            new_lines=fb_valid,
                            start_index=_approx_pos,
                            end_index=_approx_pos,
                            logger=logger,
                            ledger=ledger,
                            seg_obj=seg,
                            seen_sids=seen_sids,
                            emitted_idx_by_sid=emitted_idx_by_sid,
                            chunk_idx=idx,
                        )
                    if is_last_chunk:
                        print(
                            f"[EOD][FB_SUMMARY] After filtering: kept={len(fb_valid)} lines for last chunk.", flush=True)
                except Exception as _fe:
                    print(
                        f"[DEBUG] Fallback error ignored: {_fe}", flush=True)
                    try:
                        log_exception(
                            "parsers/openai_parser.py:convert", _fe)
                    except Exception:
                        pass
                    continue
        else:
            if any(d.get("character", "").upper() == "REJECTED" for d in fixed):
                print(
                    "[DIAG] Force-calling fallback for REJECTED-only chunk.", flush=True)
                seg_text = (ch.text or "").strip(
                ) or "REJECTED_SEGMENT_FALLBACK"
                problem_segments = [{
                    "start_idx": 0,
                    "end_idx": 0,
                    "text": seg_text,
                    "kind": "rejected"
                }]
                try:
                    _ordered_segments2 = sorted(
                        problem_segments, key=lambda s: int(s.get("start_idx", 0)))
                except Exception:
                    _ordered_segments2 = list(problem_segments)
                _running_delta2 = 0
                for seg in _ordered_segments2:
                    try:
                        try:
                            _local_speakers = []
                            for _obj in (fixed or []):
                                _ch = str((_obj or {}).get(
                                    "character", "")).strip()
                                if _ch and _ch.lower() not in ("narrator", "ambiguous", "rejected"):
                                    _local_speakers.append(_ch)
                            scoped_known = sorted(set(_local_speakers))
                            if "Narrator" not in scoped_known:
                                scoped_known.append("Narrator")
                        except Exception:
                            scoped_known = ["Narrator"]
                        fb_raw = call_frendli_fallback(
                            system_prompt,
                            seg.get("text", ""),
                            known_characters=scoped_known,
                        )
                        fb_lines = _parse_jsonl(
                            _preclean_jsonl(fb_raw))
                        try:
                            fb_lines = [_sanitize_character(
                                x, sorted(list(state.known_characters))) for x in fb_lines]
                        except Exception:
                            pass
                        fb_valid, warnings = validate_and_fix(
                            fb_lines, warnings, state, kb=parser.kb, allowed_emotions=parser.allowed_emotions, memory=parser.memory)
                        try:
                            _sidx = seg.get("start_idx", 0)
                            for _ln in fb_valid:
                                if isinstance(_ln, dict):
                                    _ln["_span_start"] = _sidx
                        except Exception:
                            pass
                        for _ln in fb_valid:
                            try:
                                _ln.setdefault("_src", "fb")
                            except Exception:
                                pass
                        try:
                            logger = None
                        except Exception:
                            logger = None
                        try:
                            _base_pos = int(seg.get("start_idx", 0))
                        except Exception:
                            _base_pos = 0
                        _approx_pos = _base_pos + _running_delta2
                        try:
                            print(
                                f"[REINJ_DELTA] base_pos={_base_pos} approx_pos={_approx_pos} delta={_running_delta2}", flush=True)
                        except Exception:
                            pass
                        if fb_valid:
                            replace_or_insert_lines(
                                dialogues=fixed,
                                new_lines=fb_valid,
                                start_index=_approx_pos,
                                end_index=_approx_pos,
                                logger=logger,
                            )
                            _running_delta2 += len(fb_valid)
                    except Exception as _fe:
                        print(
                            f"[DIAG] Forced fallback failed: {_fe}", flush=True)
                        try:
                            log_exception(
                                "parsers/openai_parser.py:convert", _fe)
                        except Exception:
                            pass
                        continue
            else:
                print("[DEBUG] No fallback needed for this chunk", flush=True)
        # F. Duplicate / relocation detection (diagnostic-only)
        try:
            sent_infos = _fb_split_sents(ch.text)
            # map normalized sentence text → sent_id for this chunk
            _sent_norm_to_id = {si.get("norm"): si.get(
                "sent_id") for si in sent_infos if si.get("norm")}
            for out_i, d in enumerate(fixed):
                norm_out = _fb_norm_cmp(d.get("text", ""))
                sid = _sent_norm_to_id.get(norm_out)
                if not sid:
                    continue
                if sid in _seen_sent_ids:
                    first_chunk, first_idx = _seen_sent_ids[sid]
                    # same sentence appears again
                    try:
                        src = (d.get("_src") or "ai")
                        diag_print(
                            f"[DUP_XLOC] sent_id={sid} first_at={first_idx} again_at={out_i} source={src}")
                        if first_chunk != idx:
                            diag_print(
                                f"[CROSS_CHUNK_MOVE] sent_id={sid} from_chunk={first_chunk} to_chunk={idx} reason=reinjection")
                    except Exception:
                        pass
                else:
                    _seen_sent_ids[sid] = (idx, out_i)
        except Exception:
            pass

        per_chunk_dialogues.append(list(fixed))
        all_dialogues.extend(fixed)
        # advance char cursor by chunk text length if no absolute offset provided
        try:
            if not hasattr(ch, "start_char") and not hasattr(ch, "abs_offset"):
                char_cursor += len(ch.text or "")
        except Exception:
            pass

    try:
        merged_parsed_lines = _dedupe_chunk_boundaries(per_chunk_dialogues)
    except Exception:
        merged_parsed_lines = list(all_dialogues)

    if parser.legacy_base_parser:
        all_dialogues = deduplicate_lines_exact(merged_parsed_lines)
    else:
        all_dialogues = deduplicate_lines(merged_parsed_lines)

    try:
        print("[PIPE] order=detect→fallback→dedup ok=True", flush=True)
        print(
            f"[ATTR_SUMMARY] phase=post_dedup { _attr_counts(all_dialogues) }", flush=True)
    except Exception:
        pass

    reconciled: List[Dict[str, Any]] = []
    for item in all_dialogues:
        if not parser.include_narration and str(item.get("character")).strip() == "Narrator":
            continue
        if reconciled and reconciled[-1]["character"] == item["character"]:
            reconciled[-1]["text"] = f"{reconciled[-1]['text']} {item['text']}".strip()
            merged_emotions = list(dict.fromkeys(
                reconciled[-1]["emotions"] + item["emotions"]))
            if len(merged_emotions) >= 2:
                merged_emotions = merged_emotions[:2]
            else:
                merged_emotions = ensure_two_emotions(
                    item["character"], merged_emotions, reconciled[-1]["text"], parser.kb, parser.allowed_emotions, parser.memory)
            reconciled[-1]["emotions"] = merged_emotions
        else:
            reconciled.append(item)

    try:
        full_text = raw_text
        if parser.legacy_base_parser:
            reconciled = _simple_reinject_missing_as_narrator(
                full_text, reconciled)
        else:
            from .fallback.detection import _split_sentences as _split_sents
            sent_infos: List[Dict[str, Any]] = _split_sents(raw_text)

            present = [_norm_for_compare(d.get("text", ""))
                       for d in reconciled]
            present_set = {p for p in present if p}

            # Prefer SID-based coverage when ledger is available
            present_sid_set: Set[str] = set()
            try:
                present_sid_set = {str(d.get("_sid")).strip()
                                   for d in reconciled if d.get("_sid")}
            except Exception:
                present_sid_set = set()
            try:
                # include those anchored during fallback reinjection
                present_sid_set |= set(seen_sids)
            except Exception:
                pass

            lnorm_to_sid: Dict[str, str] = {}
            if ledger is not None:
                try:
                    from .utils_misc import normalize_for_sid as _norm_sid
                    for entry in (ledger or []):
                        try:
                            t = _norm_sid((entry or {}).get("text", ""))
                            s = (entry or {}).get("sid")
                            if t and s:
                                lnorm_to_sid[t] = s
                        except Exception:
                            continue
                except Exception:
                    lnorm_to_sid = {}

            # If OpenAI base produced nothing, or chunk was largely rejected, force reinject
            try:
                base_valid_total = sum(1 for d in reconciled if (
                    d.get("_src") or "ai").strip() == "ai")
            except Exception:
                base_valid_total = 0
            try:
                rejected_total = sum(1 for d in reconciled if str(
                    d.get("character", "")).upper() == "REJECTED")
            except Exception:
                rejected_total = 0
            force_reinject_missing = (base_valid_total == 0) or (
                rejected_total >= max(1, len(reconciled)) * 0.5)

        def _extract_quotes_local(s: str) -> List[str]:
            qs: List[str] = []
            try:
                patterns = [
                    r'"([^"]+)' + r'"',
                    r'“([^”]+)”',
                    r'‘([^’]+)’',
                    r"'([^']+)'"
                ]
                for pat in patterns:
                    for m in _re.findall(pat, s):
                        if m and m.strip():
                            qs.append(_norm_for_compare(m))
            except Exception:
                pass
            return qs

        missing_indices: List[int] = []
        for si in sent_infos:
            s_norm = si.get("norm", "")
            if not s_norm:
                continue
            tok_count = len((s_norm or "").split())
            # SID coverage (preferred when resolvable)
            covered = False
            if ledger is not None:
                try:
                    from .utils_misc import normalize_for_sid as _norm_sid
                    sid = lnorm_to_sid.get(
                        _norm_sid(si.get("text", ""))) if 'lnorm_to_sid' in locals() else None
                    if sid and sid in present_sid_set:
                        covered = True
                except Exception:
                    covered = False
            # Very short sentences: only exact normalized match (no sim)
            if not covered and tok_count <= 3:
                if s_norm in present_set:
                    covered = True
                    if diag_enabled():
                        try:
                            diag_print(
                                f"[FINAL_COVERAGE] short_line exact-match covered: '{s_norm}'")
                        except Exception:
                            pass
                else:
                    if diag_enabled():
                        try:
                            diag_print(
                                f"[FINAL_COVERAGE] short_line sim check disabled for: '{s_norm}'")
                        except Exception:
                            pass
            # Short sentences: raise sim threshold to 0.70
            if not covered and 4 <= tok_count <= 6:
                covered = any((_token_similarity(s_norm, p) >= 0.70)
                              for p in present_set)
            # Longer sentences: default sim threshold 0.60
            if not covered and tok_count > 6:
                covered = any((_token_similarity(s_norm, p) >= 0.60)
                              for p in present_set)
            if not covered and parser.REINJECT_STRICT:
                quotes = _extract_quotes_local(si.get("text", ""))
                if quotes and any(q in present_set for q in quotes):
                    covered = True
            if not covered:
                missing_indices.append(si["index"])

        groups: List[Tuple[int, int]] = []
        if missing_indices:
            missing_indices = sorted(set(missing_indices))
            start = prev = missing_indices[0]
            for i in missing_indices[1:]:
                if i == prev + 1:
                    prev = i
                    continue
                groups.append((start, prev))
                start = prev = i
            groups.append((start, prev))

        to_reinject: List[str] = []
        for a, b in groups:
            length = b - a + 1
            sent_texts = [si["text"]
                          for si in sent_infos if a <= si["index"] <= b]
            avg_tokens = sum(len(t.split())
                             for t in sent_texts) / max(1, len(sent_texts))
        try:
            print(
                f"[REINJ_GROUP] span=({a},{b}) count={len(sent_texts)}", flush=True)
            for _txt in sent_texts:
                print(
                    f"    [REINJ_LINE] text='{(_txt or '')[:80]}'", flush=True)
        except Exception:
            pass
        effective_strict = bool(
            parser.REINJECT_STRICT and not force_reinject_missing)
        if (not effective_strict) or length >= 2 or avg_tokens <= 12:
            if effective_strict and length < 2 and avg_tokens <= 12:
                try:
                    print(
                        f"[DIAG][REINJECT] single-line reinjection enabled (avg_tokens={avg_tokens:.1f}) span=[{a}:{b}]", flush=True)
                except Exception:
                    pass
            for si in sent_infos:
                if a <= si["index"] <= b:
                    to_reinject.append(si["text"])

        _diag_reinj_considered_total += len(sent_infos)
        try:
            print(
                f"[DIAG] Reinjection considered sentences: {len(sent_infos)}", flush=True)
            print(
                f"[DIAG] Reinjection: will add {len(to_reinject)} sentence(s)", flush=True)
        except Exception:
            pass
        for m in to_reinject:
            try:
                last_char = reconciled[-1]["character"] if reconciled else "Narrator"
            except Exception:
                last_char = "Narrator"
            txt_lower = (m or "").lower()
            char_guess = last_char
            try:
                if any(w in txt_lower for w in ['\"', '"', " said", "asked", "whispered", "replied", "murmured", "moaned", "commanded"]) and last_char not in ("Narrator", "Ambiguous"):
                    char_guess = last_char
                elif (m or "").strip().startswith(('\"', '"')):
                    char_guess = "Ambiguous" if last_char in (
                        "Narrator", "Ambiguous") else last_char
                else:
                    char_guess = "Narrator"
            except Exception:
                char_guess = last_char or "Narrator"
            reconciled.append({
                "character": char_guess,
                "emotions": [],
                "text": m,
                "id": f"reinjected-{abs(hash(m))}",
                "_src": "reinj",
            })
            try:
                reason = "prev" if char_guess not in ("Narrator", "Ambiguous") else (
                    "quote-no-speaker" if (m or "").strip().startswith(('\"', '"')) else "none")
                print(
                    f"[REINJ_FIX] text='{(m or '')[:80]}' char_guess={char_guess} reason={reason}",
                    flush=True,
                )
            except Exception:
                pass
        _diag_reinj_added_total += len(to_reinject)
        if to_reinject:
            warnings.append(
                f"Re-injected {len(to_reinject)} missing sentence(s) as Narrator.")
    except Exception as _se:
        try:
            log_exception("parsers/openai_parser.py:convert", _se)
        except Exception:
            pass
        pass

    if True:
        try:
            tail = reconciled[-20:] if len(
                reconciled) > 20 else reconciled[:]
            print(
                "\n===== [EOD][TAIL] FINAL RECONCILED LAST 20 LINES (pre-REJECTED-purge) =====", flush=True)
            for k, d in enumerate(tail, start=max(1, len(reconciled)-len(tail)+1)):
                ch_name = d.get('character', '')
                txt = (d.get('text', '') or '')[:180]
                src = d.get('_src', '?')
                print(f"{k:03d}. [{src}] {ch_name}: {txt}", flush=True)
            try:
                narr_reinj = sum(1 for d in tail if str(d.get('character', '')).strip(
                ) == 'Narrator' and d.get('_src') in {'reinj', 'fb'})
                if narr_reinj >= 10 and len(tail) >= 10:
                    print(
                        f"[TAIL_CHECK] suspicious_tail narrator_reinj={narr_reinj} last20_total={len(tail)}", flush=True)

                def norm(s): return (s or '').strip().lower()
                earlier = {norm(x.get('text', ''))
                           for x in reconciled[:-20]}
                dup_tail = sum(1 for d in tail if norm(
                    d.get('text', '')) in earlier)
                if dup_tail:
                    print(
                        f"[TAIL_CHECK] dup_in_tail count={dup_tail}", flush=True)
            except Exception:
                pass
            print("===== [EOD][TAIL] END =====\n", flush=True)
        except Exception:
            print("[EOD] <failed to print reconciled tail>", flush=True)

    try:
        before = len(reconciled)
        reconciled = [d for d in reconciled if str(
            d.get("character", "")).upper() != "REJECTED"]
        purged = before - len(reconciled)
        if purged:
            print(
                f"[DIAG] Purged REJECTED lines before formatting: {purged}", flush=True)
    except Exception:
        pass

    formatted_lines: List[str] = []
    for d in reconciled:
        em_text = "".join([f"({e})" for e in d.get("emotions", [])])
        formatted_lines.append(
            f"[{d['character']}] {em_text}: {d['text']}".strip())

    try:
        for i, d in enumerate(reconciled[-20:]):
            print(
                f"[ORDER] idx={i:04d} src={d.get('_src')} char={d.get('character')} text='{(d.get('text') or '')[:60]}'",
                flush=True,
            )
    except Exception:
        pass

    # [CHUNK_SUMMARY] emit summary per chunk before final write (using diagnostics context)
    try:
        from .fallback.diag_ctx import _get_tail_appends as _fb_get_tail
        for i, chunk_lines in enumerate(per_chunk_dialogues):
            if not diag_enabled():
                break
            src_counts = {"openai": 0, "friendli": 0,
                          "fallback": 0, "duplicates": 0}
            seen_ids = set()
            for d in chunk_lines:
                src = str(d.get("_src") or "ai")
                if src == "ai":
                    src_counts["openai"] += 1
                elif src == "fb":
                    src_counts["friendli"] += 1
                elif src == "reinj":
                    src_counts["fallback"] += 1
                # light duplicate counter by normalized text
                nid = (d.get("character", "").strip().lower(),
                       (d.get("text", "") or "").strip().lower())
                if nid in seen_ids:
                    src_counts["duplicates"] += 1
                else:
                    seen_ids.add(nid)
            tail_n = _fb_get_tail(i)
            diag_print(
                f"[CHUNK_SUMMARY] chunk_idx={i} lines={len(chunk_lines)} openai={src_counts['openai']} friendli={src_counts['friendli']} fallback={src_counts['fallback']} duplicates={src_counts['duplicates']} tail_appends={tail_n}"
            )
    except Exception:
        pass

    if ledger is not None and diag_enabled():
        try:
            diag_print(f"[LEDGER_SUMMARY] count={len(ledger)}")
        except Exception:
            pass

    stats = {
        "quotes_found": len(reconciled),
        "lines_emitted": len(formatted_lines),
        "narration_blocks": sum(1 for d in reconciled if d.get("character") == "Narrator"),
    }

    try:
        print("[DIAG] ===== DIAG REPORT (batch) =====", flush=True)
        print(f"[DIAG] Fallback calls: {len(_diag_fb_calls)}", flush=True)
        for _r in _diag_fb_calls:
            print(
                f"[DIAG]   chunk={_r['chunk']} seg={_r['seg']} text_len={_r['text_len']} parsed={_r['parsed']} unknown={_r['unknown']}", flush=True)
        print(
            f"[DIAG] Reinjection totals: considered={_diag_reinj_considered_total} added={_diag_reinj_added_total}", flush=True)
        try:
            _conf = diag_consume_dedup_conflicts()
        except Exception:
            _conf = []
        print(
            f"[DIAG] Cross-speaker duplicate conflicts: {len(_conf)}", flush=True)
        for _i, (_txt, _a, _b) in enumerate((_conf or [])[:3], start=1):
            print(
                f"[DIAG]   #{_i}: '{_txt[:80]}' A={_a} B={_b}", flush=True)
        try:
            last_fb_calls = [r for r in (
                _diag_fb_calls or []) if r.get("chunk") == len(chunks)]
            _last_inserted = sum(int(r.get("parsed") or 0)
                                 for r in last_fb_calls)
            _last_unknown = sum(int(r.get("unknown") or 0)
                                for r in last_fb_calls)
            print(
                f"[EOD][SUMMARY] last_chunk_fallback_calls={len(last_fb_calls)} inserted={_last_inserted} unknown_speakers={_last_unknown}", flush=True)
        except Exception:
            pass
        print("[DIAG] ===============================", flush=True)
    except Exception:
        pass

    try:
        log_to_session("INFO", "Parser convert end (success)",
                       src="parsers/openai_parser.py:convert")
    except Exception:
        pass
    try:
        sys.stdout = _original_stdout
        sys.stderr = _original_stderr
        with open(log_path, "w", encoding="utf-8") as _f:
            _f.write(_dual_logger.get_value())
        print(
            f"[LOG_END] Full parser debug log saved to {log_path}", flush=True)
    except Exception as _le:
        try:
            _original_stdout.write(
                f"[LOG_ERROR] Failed to save debug log: {_le}\n")
        except Exception:
            pass
    return RawParseResult("\n".join(formatted_lines), reconciled, stats, ambiguities, warnings, errors)


if __name__ == "__main__":
    import os as _os
    if (_os.getenv("DEBUG_PARSER_DIAG") or "0").lower() in {"1", "true", "yes", "on"}:
        sample_text = (
            '"I can help you," she said. '
            '\nHe looked away, unsure.'
            '\n"…?" Aleksandr murmured'
            '\nShe gasped as his hands cupped her breasts.'
        )
        from .openai_parser import OpenAIParser as _Parser
        p = _Parser()
        try:
            res = p.convert(sample_text)
            print("[HARNESS] done. see diagnostics above.")
        except Exception as e:
            print(f"[HARNESS_ERR] {e}")
