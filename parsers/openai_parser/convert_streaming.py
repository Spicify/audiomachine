from __future__ import annotations

from typing import Any, Dict, Iterator, List, Set
from pathlib import Path
import datetime
import sys
import time
import hashlib

from .core_types import ParserState, DualLogger, RawParseResult, _hash_key, _token_similarity
from .prompt_builder import build_system_prompt, build_user_prompt
from .chunker import build_chunks
from .fallback_utils import (
    detect_missing_or_rejected_lines,
    call_frendli_fallback,
    replace_or_insert_lines,
)
from .fallback_utils import _sanitize_character
from .openai_client import call_openai_safe, _save_debug_output
from .validator import validate_and_fix
from .utils_misc import _simple_reinject_missing_as_narrator, _build_sentence_to_pos_map, _preclean_jsonl
from utils.text_normalizer import normalize_text as _norm_for_compare


def convert_stream(parser, raw_text: str) -> Iterator[Dict]:
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

    warnings: List[str] = []
    errors: List[str] = []
    if not raw_text or not raw_text.strip():
        yield {"chunk_index": 0, "total_chunks": 0, "dialogues": [], "ambiguities": [], "warnings": warnings}
        return

    print(f">>> Building chunks (input chars={len(raw_text)})", flush=True)
    chunks = build_chunks(
        raw_text, max_tokens=parser.max_tokens_per_chunk, model=parser.model, overlap_sentences=2)
    print(f">>> Built {len(chunks)} chunks", flush=True)
    parser._run_call_durations = []
    parser._run_chunk_token_counts = []
    state = ParserState(known_characters=set(
        parser.default_known_characters))
    seen_keys: Set[str] = set()

    total = len(chunks)
    try:
        for idx, ch in enumerate(chunks):
            print(
                f">>> Starting chunk {idx+1}/{total} (approx tokens={ch.token_count})", flush=True)
            summary = {
                "recent_characters": list(state.known_characters)[:20],
                "last_speaker": state.last_speaker,
                "last_emotions": {k: v[-2:] for k, v in state.last_emotions.items()},
                "unresolved": [a.get("text", "") for a in state.unresolved_ambiguities][-5:],
            }
            system_prompt = build_system_prompt(parser.allowed_emotions, list(
                state.known_characters), parser.include_narration, summary)
            user_prompt = build_user_prompt(ch.text, None)

            cache_key = _hash_key(ch.text, summary)
            chunk_duration_sec = 0.0
            if cache_key in parser._cache:
                print(">>> Using cached response", flush=True)
                raw_output = parser._cache[cache_key]
                chunk_duration_sec = 0.0
            else:
                try:
                    _t0 = time.monotonic()
                    raw_output = call_openai_safe(
                        system_prompt, user_prompt, client=parser.client, model=parser.model)
                    elapsed = time.monotonic() - _t0
                    parser._last_call_elapsed_sec = elapsed
                    chunk_duration_sec += elapsed
                except Exception as e:
                    errors.append(
                        f"Parser API error on chunk {idx+1}/{total}: {e}")
                    yield {"chunk_index": idx+1, "total_chunks": total, "dialogues": [], "ambiguities": [], "warnings": warnings, "errors": errors}
                    raise
                if parser.debug_save:
                    _save_debug_output(
                        raw_output, suffix=f"_chunk{idx+1}", debug_save=parser.debug_save)
                parser._cache[cache_key] = raw_output

            print(
                f">>> Got response (chars={len(raw_output)})", flush=True)
            if idx == total - 1:
                print(
                    "\n===== [EOD][CHUNK_TEXT] LAST CHUNK INPUT BEGIN =====", flush=True)
                try:
                    print(ch.text, flush=True)
                except Exception:
                    print("[EOD] <failed to print chunk text>", flush=True)
                print(
                    "===== [EOD][CHUNK_TEXT] LAST CHUNK INPUT END =====\n", flush=True)
                print(
                    "\n===== [EOD][OPENAI_RAW] LAST CHUNK RAW BEGIN (first 60 lines) =====", flush=True)
                try:
                    for i, ln in enumerate((raw_output or "").splitlines()[:60], start=1):
                        print(f"{i:02d}: {ln}", flush=True)
                except Exception:
                    print("[EOD] <failed to print raw output>", flush=True)
                print(
                    "===== [EOD][OPENAI_RAW] LAST CHUNK RAW END =====\n", flush=True)

            def _parse_jsonl(raw_output: str) -> List[Dict[str, Any]]:
                lines = [ln.strip()
                         for ln in raw_output.split("\n") if ln.strip()]
                objs: List[Dict[str, Any]] = []
                for ln in lines:
                    if not (ln.startswith("{") and ln.endswith("}")):
                        continue
                    try:
                        import json as _json
                        obj = _json.loads(ln)
                        if isinstance(obj, dict):
                            objs.append(obj)
                    except Exception:
                        continue
                return objs

            items = _parse_jsonl(raw_output)
            print(f">>> Parsed {len(items)} items", flush=True)
            if not items:
                print(">>> Retry with JSONL ONLY", flush=True)
                raw_output = call_openai_safe(
                    system_prompt + "\nIMPORTANT: Output JSONL ONLY.", user_prompt, client=parser.client, model=parser.model)
                chunk_duration_sec += parser._last_call_elapsed_sec
                if parser.debug_save:
                    _save_debug_output(
                        raw_output, suffix=f"_retry_chunk{idx+1}", debug_save=parser.debug_save)
                items = _parse_jsonl(raw_output)
                print(
                    f">>> Parsed after retry: {len(items)} items", flush=True)

            chunk_ambs: List[Dict[str, Any]] = []
            for it in items:
                if str(it.get("character", "")).strip().lower() == "ambiguous":
                    txt = (it.get("text") or "").strip()
                    cands = [str(c).strip() for c in (
                        it.get("candidates") or []) if str(c).strip()]
                    if not cands:
                        cands = ["Unknown"]
                    chunk_ambs.append({
                        "id": f"amb-{abs(hash(txt))}",
                        "text": txt,
                        "candidates": cands[:5],
                    })

            fixed, warnings = validate_and_fix(
                items, warnings, state, kb=parser.kb, allowed_emotions=parser.allowed_emotions, memory=parser.memory)
            for _ln in fixed:
                try:
                    _ln.setdefault("_src", "ai")
                except Exception:
                    pass

            try:
                problem_segments = detect_missing_or_rejected_lines(
                    ch.text, fixed)
                print(
                    f"[DEBUG] Fallback detection: {len(problem_segments)} segment(s) found", flush=True)
                if not problem_segments and any(
                    d.get("character") == "REJECTED" for d in fixed
                ):
                    print(
                        "[DIAG] WARNING: REJECTED present but detection found 0 segments.", flush=True)
                is_last_chunk = (idx == total - 1)
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
                        merged_text = " ".join(
                            seg.get("text", "") for seg in problem_segments[:15])
                        problem_segments = [{"text": merged_text}]
                        parser._eod_fallback_throttled = True
                    else:
                        parser._eod_fallback_throttled = False
            except Exception as e:
                problem_segments = []
                try:
                    import traceback as _tb
                    print(
                        "[DIAG][ERROR] (streaming) detect_missing_or_rejected_lines() crashed:",
                        type(e).__name__, str(e), flush=True,
                    )
                    _tb.print_exc()
                except Exception:
                    pass
            if problem_segments:
                try:
                    _ordered_segments3 = sorted(
                        problem_segments, key=lambda s: int(s.get("start_idx", 0)))
                except Exception:
                    _ordered_segments3 = list(problem_segments)
                _running_delta3 = 0
                for _seg_i, seg in enumerate(_ordered_segments3, start=1):
                    try:
                        if idx == total - 1:
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
                        if idx == total - 1:
                            print(
                                "----- [EOD][FB] RAW Friendli OUT (first 30 lines) -----", flush=True)
                            try:
                                for j, ln in enumerate((fb_raw or '').splitlines()[:30], start=1):
                                    print(f"{j:02d}: {ln}", flush=True)
                            except Exception:
                                print(
                                    "[EOD] <failed to print friendli raw>", flush=True)
                        try:
                            _fb_lines_preview = (
                                fb_raw or "").splitlines()[:2]
                            print(
                                f"[DIAG] Fallback RAW (first 2 lines): {_fb_lines_preview}", flush=True)
                        except Exception:
                            pass
                        if parser.debug_save:
                            try:
                                Path("debug_outputs").mkdir(exist_ok=True)
                                ts2 = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                                fname = f"fallback_raw_{ts2}_chunk{idx+1}_seg{_seg_i}.txt"
                                fpath = Path("debug_outputs") / fname
                                with open(fpath, "w", encoding="utf-8") as _f:
                                    _f.write(fb_raw or "")
                                print(
                                    f"[DIAG] Saved fallback RAW to {fpath}", flush=True)
                            except Exception:
                                pass

                        def _parse_jsonl(raw_output: str) -> List[Dict[str, Any]]:
                            lines = [ln.strip()
                                     for ln in raw_output.split("\n") if ln.strip()]
                            objs: List[Dict[str, Any]] = []
                            for ln in lines:
                                if not (ln.startswith("{") and ln.endswith("}")):
                                    continue
                                try:
                                    import json as _json
                                    obj = _json.loads(ln)
                                    if isinstance(obj, dict):
                                        objs.append(obj)
                                except Exception:
                                    continue
                            return objs

                        fb_lines = _parse_jsonl(
                            _preclean_jsonl(fb_raw))
                        try:
                            fb_lines = [_sanitize_character(
                                x, sorted(list(state.known_characters))) for x in fb_lines]
                        except Exception:
                            pass
                        print(
                            f"[DEBUG] Fallback parsed {len(fb_lines)} line(s)", flush=True)
                        fb_valid, warnings = validate_and_fix(
                            fb_lines, warnings, state, kb=parser.kb, allowed_emotions=parser.allowed_emotions, memory=parser.memory)
                        try:
                            approx = seg.get("_approx_indices") or []
                            if approx and fb_valid:
                                for k, _ln in enumerate(fb_valid):
                                    _ln["_span_start"] = int(
                                        approx[min(k, len(approx)-1)])
                        except Exception:
                            pass
                        segment_text = seg.get("text", "") or ""
                        from .fallback_utils import filter_fallback_lines as _fb_filter
                        fb_valid = _fb_filter(segment_text, fb_valid)
                        print(
                            f"[EOD][FB_SUMMARY] After fuzzy filter: kept={len(fb_valid)}", flush=True)
                        for _ln in fb_valid:
                            try:
                                _ln.setdefault("_src", "fb")
                            except Exception:
                                pass
                        if idx == total - 1:
                            print(
                                f"[EOD][FB] validated_and_kept={len(fb_valid)} line(s) for seg#{_seg_i}", flush=True)
                        print(
                            f"[DEBUG] Fallback validated {len(fb_valid)} line(s)", flush=True)
                        try:
                            allowed = set(state.known_characters) | {
                                "Narrator", "Ambiguous"}
                            for _ln in fb_valid:
                                ch_name = str(_ln.get("character", ""))
                                if ch_name and ch_name not in allowed:
                                    _txt = (_ln.get("text", "") or "")[:80]
                                    print(
                                        f"[DIAG][UNKNOWN_SPEAKER] '{ch_name}' text='{_txt}' (allowed_size={len(allowed)})", flush=True)
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
                        _approx_pos = _base_pos + _running_delta3
                        try:
                            print(
                                f"[REINJ_DELTA] base_pos={_base_pos} approx_pos={_approx_pos} delta={_running_delta3}", flush=True)
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
                            _running_delta3 += len(fb_valid)
                        if is_last_chunk:
                            print(
                                f"[EOD][FB_SUMMARY] After filtering: kept={len(fb_valid)} lines for last chunk.", flush=True)
                    except Exception as _fe:
                        print(
                            f"[DEBUG] Fallback error ignored: {_fe}", flush=True)
                        continue
            else:
                if any(d.get("character", "").upper() == "REJECTED" for d in fixed):
                    if getattr(parser, "_eod_fallback_throttled", False):
                        print(
                            "[INFO][FB] Skipping forced fallback; throttle already engaged.", flush=True)
                        return
                    print(
                        "[DIAG] Force-calling fallback for REJECTED-only chunk (streaming).", flush=True)
                    seg_text = (ch.text or "").strip(
                    ) or "REJECTED_SEGMENT_FALLBACK"
                    problem_segments = [{
                        "start_idx": 0,
                        "end_idx": 0,
                        "text": seg_text,
                        "kind": "rejected"
                    }]
                    for seg in problem_segments:
                        try:
                            fb_raw = call_frendli_fallback(
                                system_prompt,
                                seg.get("text", ""),
                                known_characters=list(
                                    state.known_characters or []),
                            )

                            def _parse_jsonl(raw_output: str) -> List[Dict[str, Any]]:
                                lines = [ln.strip() for ln in raw_output.split(
                                    "\n") if ln.strip()]
                                objs: List[Dict[str, Any]] = []
                                for ln in lines:
                                    if not (ln.startswith("{") and ln.endswith("}")):
                                        continue
                                    try:
                                        import json as _json
                                        obj = _json.loads(ln)
                                        if isinstance(obj, dict):
                                            objs.append(obj)
                                    except Exception:
                                        continue
                                return objs

                            fb_lines = _parse_jsonl(
                                _preclean_jsonl(fb_raw))
                            fb_valid, warnings = validate_and_fix(
                                fb_lines, warnings, state, kb=parser.kb, allowed_emotions=parser.allowed_emotions, memory=parser.memory)
                            try:
                                logger = None
                            except Exception:
                                logger = None
                            replace_or_insert_lines(
                                dialogues=fixed,
                                new_lines=fb_valid,
                                start_index=seg.get("start_idx", 0),
                                end_index=seg.get(
                                    "end_idx", seg.get("start_idx", 0)),
                                logger=logger,
                            )
                        except Exception as _fe:
                            print(
                                f"[DIAG] Forced fallback failed (streaming): {_fe}", flush=True)
                            continue
                else:
                    print(
                        "[DEBUG] No fallback needed for this chunk", flush=True)

            filtered: List[Dict[str, Any]] = []
            for it in fixed:
                key = hashlib.sha256(
                    f"{(it.get('character') or '').strip().lower()}|{(it.get('text') or '').strip()}".encode("utf-8")).hexdigest()
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                filtered.append(it)

            try:
                if parser.legacy_base_parser:
                    filtered = _simple_reinject_missing_as_narrator(
                        ch.text, filtered)
                else:
                    from .fallback_utils import _split_sentences as _split_sents
                    sent_infos = _split_sents(ch.text)
                    present = {_norm_for_compare(
                        d.get("text", "")) for d in filtered}
                    reinjected = 0
                    print(
                        f"[DIAG] Streaming reinjection: considered={len(sent_infos)}", flush=True)

                def _extract_quotes_local(s: str) -> List[str]:
                    qs: List[str] = []
                    try:
                        patterns = [
                            r'"([^"]+)' + r'"',
                            r'“([^”]+)”',
                            r'‘([^’]+)’',
                            r"'([^']+)'"
                        ]
                        import re as __re
                        for pat in patterns:
                            for m in __re.findall(pat, s):
                                if m and m.strip():
                                    qs.append(_norm_for_compare(m))
                    except Exception:
                        pass
                    return qs

                missing_idx: list[int] = []
                for si in sent_infos:
                    s_norm = si.get("norm", "")
                    if not s_norm:
                        continue
                    covered = (s_norm in present) or any(
                        (s_norm in p or p in s_norm) for p in present)
                    if not covered and parser.REINJECT_STRICT:
                        qs = _extract_quotes_local(si.get("text", ""))
                        if qs and any(q in present for q in qs):
                            covered = True
                    if not covered:
                        missing_idx.append(si["index"])
                groups: list[tuple[int, int]] = []
                if missing_idx:
                    missing_idx = sorted(set(missing_idx))
                    start = prev = missing_idx[0]
                    for i in missing_idx[1:]:
                        if i == prev + 1:
                            prev = i
                            continue
                        groups.append((start, prev))
                        start = prev = i
                    groups.append((start, prev))
                for a, b in groups:
                    length = b - a + 1
                    sent_texts = [si["text"]
                                  for si in sent_infos if a <= si["index"] <= b]
                    avg_tokens = sum(len(t.split())
                                     for t in sent_texts) / max(1, len(sent_texts))
                    if (not parser.REINJECT_STRICT) or length >= 2 or avg_tokens <= 12:
                        if parser.REINJECT_STRICT and length < 2 and avg_tokens <= 12:
                            try:
                                print(
                                    f"[DIAG][REINJECT] (streaming) single-line reinjection enabled (avg_tokens={avg_tokens:.1f}) span=[{a}:{b}]", flush=True)
                            except Exception:
                                pass
                        for si in sent_infos:
                            if a <= si["index"] <= b:
                                filtered.append({
                                    "character": "Narrator",
                                    "emotions": [],
                                    "text": si["text"],
                                    "id": f"reinjected-{abs(hash(si['text']))}",
                                    "_src": "reinj",
                                })
                                reinjected += 1
                                try:
                                    _samples = [d.get("text", "")
                                                for d in filtered[:2]]
                                    print(
                                        f"[DIAG] Streaming reinjected (strict): '{(si['text'] or '')[:120]}' | samples={_samples}", flush=True)
                                except Exception:
                                    pass
                if reinjected:
                    warnings.append(
                        f"Streaming: re-injected {reinjected} missing sentence(s) as Narrator.")
            except Exception:
                pass
            try:
                _purge_count = sum(1 for d in filtered if str(
                    d.get("character", "")).upper() == "REJECTED")
                if _purge_count:
                    print(
                        f"[DIAG] (streaming) Purged {_purge_count} residual REJECTED lines.", flush=True)
                filtered = [d for d in filtered if str(
                    d.get("character", "")).upper() != "REJECTED"]
            except Exception:
                pass

            if idx == total - 1:
                try:
                    tail = filtered[-20:] if len(
                        filtered) > 20 else filtered[:]
                    print(
                        "\n===== [EOD][TAIL] FINAL RECONCILED LAST 20 LINES (pre-REJECTED-purge/stream) =====", flush=True)
                    for k, d in enumerate(tail, start=max(1, len(filtered)-len(tail)+1)):
                        ch_name = d.get('character', '')
                        txt = (d.get('text', '') or '')[:180]
                        src = d.get('_src', '?')
                        print(
                            f"{k:03d}. [{src}] {ch_name}: {txt}", flush=True)
                    print("===== [EOD][TAIL] END =====\n", flush=True)
                except Exception:
                    print("[EOD] <failed to print reconciled tail>", flush=True)
            print(
                f">>> Yielding chunk {idx+1}/{total}: dialogues={len(filtered)}, ambiguities={len(chunk_ambs)}", flush=True)
            parser._run_call_durations.append(chunk_duration_sec)
            parser._run_chunk_token_counts.append(
                int(getattr(ch, "token_count", 0) or 0))
            try:
                if not filtered:
                    print(
                        f"[WARN][STREAM] No lines for chunk {idx+1}/{total}",
                        flush=True,
                    )
            except Exception:
                pass
            yield {
                "chunk_index": idx + 1,
                "total_chunks": total,
                "dialogues": filtered,
                "ambiguities": chunk_ambs,
                "warnings": warnings,
            }
    finally:
        try:
            log_path2 = Path(__file__).parent / "parser_logs.txt"
            entries: List[str] = []
            for i, d in enumerate(parser._run_call_durations or []):
                tok = 0
                if i < len(parser._run_chunk_token_counts):
                    tok = int(parser._run_chunk_token_counts[i] or 0)
                entries.append(f"{d:.2f} ({tok})")
            line = ", ".join(entries)
            with open(log_path2, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            print(f">>> Wrote parser timings/tokens: [{line}]", flush=True)
        except Exception as _e:
            print(f">>> Failed to write parser timings: {_e}", flush=True)
        try:
            sys.stdout = _original_stdout
            sys.stderr = _original_stderr
            with open(Path("logs") / f"parser_run_{ts}.txt", "w", encoding="utf-8") as _f:
                _f.write(_dual_logger.get_value())
            print(
                f"[LOG_END] Full parser debug log saved to logs/parser_run_{ts}.txt", flush=True)
        except Exception as _le:
            try:
                _original_stdout.write(
                    f"[LOG_ERROR] Failed to save debug log: {_le}\n")
            except Exception:
                pass
