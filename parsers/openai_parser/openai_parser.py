from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import time

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field, ValidationError

from audio.utils import get_flat_emotion_tags, get_flat_character_voices
from settings import OPENAI_API_KEY
from .chunker import build_chunks, deduplicate_lines
from .chunker import diag_consume_dedup_conflicts
from .emotion_utils import EmotionMemory, build_emotion_kb, ensure_two_emotions, get_allowed_emotions
from .prompt_builder import build_system_prompt, build_user_prompt
from .fallback_utils import (
    detect_missing_or_rejected_lines,
    call_frendli_fallback,
    replace_or_insert_lines,
)
from utils.log_instrumentation import log_timed_action
from utils.session_logger import log_to_session, log_exception


@dataclass
class RawParseResult:
    formatted_text: str
    dialogues: List[Dict]
    stats: Dict[str, int]
    ambiguities: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class DialogueLine(BaseModel):
    character: str
    emotions: List[str] = Field(min_length=2, max_length=2)
    text: str
    candidates: Optional[List[str]] = None


class ParserState(BaseModel):
    known_characters: Set[str] = Field(default_factory=set)
    last_speaker: Optional[str] = None
    last_emotions: Dict[str, List[str]] = Field(default_factory=dict)
    unresolved_ambiguities: List[Dict[str, Any]] = Field(default_factory=list)


def _hash_key(text: str, state: Dict[str, Any]) -> str:
    payload = json.dumps({"t": text, "s": state},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class OpenAIParser:
    # Reinjection strictness gate
    REINJECT_STRICT: bool = True

    def __init__(
        self,
        model: str = "gpt-5-mini",
        include_narration: bool = True,
        max_tokens_per_chunk: int = 1500,
        debug_save: bool = False,
    ):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        self.include_narration = include_narration
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.debug_save = debug_save
        self.allowed_emotions = get_allowed_emotions()
        self.default_known_characters = set(get_flat_character_voices().keys())
        self.kb = build_emotion_kb()
        self.memory = EmotionMemory()
        self._cache: Dict[str, str] = {}
        # Per-run timing metrics (seconds per OpenAI call per chunk)
        self._run_call_durations: List[float] = []
        # Per-run token counts per chunk (from chunker)
        self._run_chunk_token_counts: List[int] = []
        self._last_call_elapsed_sec: float = 0.0

    def _state_summary(self, state: ParserState) -> Dict[str, Any]:
        return {
            "recent_characters": list(state.known_characters)[:20],
            "last_speaker": state.last_speaker,
            "last_emotions": {k: v[-2:] for k, v in state.last_emotions.items()},
            "unresolved": [a.get("text", "") for a in state.unresolved_ambiguities][-5:],
        }

    def _save_debug_output(self, raw_text: str, suffix: str = "") -> None:
        if not self.debug_save:
            return
        try:
            Path("debug_outputs").mkdir(exist_ok=True)
            ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            name = f"openai_parser_{ts}{suffix}.txt"
            with open(Path("debug_outputs") / name, "w", encoding="utf-8") as f:
                f.write(raw_text)
        except Exception:
            pass

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    @log_timed_action("OpenAI call duration")
    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        start_time = time.monotonic()
        print(">>> Calling OpenAI…", flush=True)
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        out = (response.output_text or "").strip()
        self._last_call_elapsed_sec = (time.monotonic() - start_time)
        print(
            f">>> OpenAI responded in {self._last_call_elapsed_sec*1000.0:.0f} ms (chars={len(out)})", flush=True)
        return out

    def _validate_and_fix(self, items: List[Dict[str, Any]], warnings: List[str], state: ParserState) -> Tuple[List[Dict[str, Any]], List[str]]:
        result: List[Dict[str, Any]] = []
        for it in items:
            # [DIAG] Preserve REJECTED lines end-to-end so detection can trigger fallback
            try:
                if str(it.get("character", "")).upper() == "REJECTED":
                    print("[DIAG] REJECTED line entering validator:",
                          (it.get("text", "") or "")[:60], flush=True)
                    result.append(it)
                    continue
            except Exception:
                pass
            char = (it.get("character") or "").strip()
            txt = (it.get("text") or "").strip()
            ems = it.get("emotions") or []

            # ensure 2 canonical emotions
            ems = ensure_two_emotions(
                char, ems, txt, self.kb, self.allowed_emotions, self.memory)

            fixed = {"character": char, "text": txt, "emotions": ems}
            if str(char).lower() == "ambiguous":
                # Attach candidates and a stable id for UI resolution
                if it.get("candidates"):
                    cands = [str(c).strip() for c in (
                        it.get("candidates") or []) if str(c).strip()]
                    if cands:
                        fixed["candidates"] = cands[:5]
                fixed["id"] = f"amb-{abs(hash(txt))}"

            try:
                DialogueLine(**fixed)
            except ValidationError as ve:
                warnings.append(
                    f"Validation dropped line: {fixed} ({ve.errors()[:1]})")
                continue

            # update memory and continuity
            if fixed["character"].lower() != "narrator" and fixed["character"].lower() != "ambiguous":
                state.known_characters.add(fixed["character"])
                state.last_speaker = fixed["character"]
                state.last_emotions.setdefault(
                    fixed["character"], []).extend(fixed["emotions"])
                self.memory.push(fixed["character"], fixed["emotions"])

            result.append(fixed)
        return result, warnings

    def _parse_jsonl(self, raw_output: str) -> List[Dict[str, Any]]:
        lines = [ln.strip() for ln in raw_output.split("\n") if ln.strip()]
        objs: List[Dict[str, Any]] = []
        for ln in lines:
            if not (ln.startswith("{") and ln.endswith("}")):
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    objs.append(obj)
            except Exception:
                continue
        return objs

    # Friendli-only pre-clean for JSONL quirks
    def _preclean_jsonl(self, raw: str) -> str:
        txt = (raw or "").strip()
        if not txt:
            return ""
        # strip code fences
        try:
            if (txt.startswith("```") and txt.endswith("```")) or (txt.startswith("~~~") and txt.endswith("~~~")):
                txt = txt.strip("`~\n ")
        except Exception:
            pass
        # JSON array → emit as JSONL
        try:
            import json as _json
            import re as _re
            if re.match(r"^\s*\[", txt):
                arr = _json.loads(txt)
                if isinstance(arr, list):
                    return "\n".join(_json.dumps(obj, ensure_ascii=False) for obj in arr if isinstance(obj, dict))
        except Exception:
            pass
        # Extract top-level {...} blocks naively
        try:
            import re as _re
            blocks = _re.findall(r"\{[\s\S]*?\}", txt, _re.DOTALL)
            if blocks:
                # remove trailing commas at line ends
                cleaned_lines = []
                for b in blocks:
                    b2 = _re.sub(r",\s*$", "", b.strip(), flags=_re.MULTILINE)
                    cleaned_lines.append(b2)
                return "\n".join(cleaned_lines)
        except Exception:
            pass
        return txt

    def convert(self, raw_text: str) -> RawParseResult:
        try:
            log_to_session(
                "INFO", f"Parser convert start (chars={len(raw_text or '')})", src="parsers/openai_parser.py:convert")
        except Exception:
            pass
        warnings: List[str] = []
        errors: List[str] = []
        # [DIAG] accumulators for this run
        _diag_fb_calls: List[Dict[str, Any]] = []
        _diag_reinj_considered_total: int = 0
        _diag_reinj_added_total: int = 0
        if not raw_text or not raw_text.strip():
            return RawParseResult(formatted_text="", dialogues=[], stats={}, ambiguities=[], warnings=warnings, errors=errors)

        chunks = build_chunks(
            raw_text, max_tokens=self.max_tokens_per_chunk, model=self.model, overlap_sentences=2)
        state = ParserState(known_characters=set(
            self.default_known_characters))
        all_dialogues: List[Dict[str, Any]] = []
        ambiguities: List[Dict[str, Any]] = []

        for idx, ch in enumerate(chunks):
            summary = self._state_summary(state)
            system_prompt = build_system_prompt(self.allowed_emotions, list(
                state.known_characters), self.include_narration, summary)
            user_prompt = build_user_prompt(ch.text, None)

            cache_key = _hash_key(ch.text, summary)
            if cache_key in self._cache:
                raw_output = self._cache[cache_key]
            else:
                try:
                    raw_output = self._call_openai(system_prompt, user_prompt)
                except Exception as e:
                    errors.append(
                        f"Parser API error on chunk {idx+1}/{len(chunks)}: {e}")
                    try:
                        log_exception("parsers/openai_parser.py:convert", e)
                    except Exception:
                        pass
                    raise
                if self.debug_save:
                    self._save_debug_output(
                        raw_output, suffix=f"_chunk{idx+1}")
                self._cache[cache_key] = raw_output

            items = self._parse_jsonl(raw_output)
            if not items:
                # Retry once with stricter reminder
                raw_output = self._call_openai(
                    system_prompt + "\nIMPORTANT: Output JSONL ONLY.", user_prompt)
                if self.debug_save:
                    self._save_debug_output(
                        raw_output, suffix=f"_retry_chunk{idx+1}")
                items = self._parse_jsonl(raw_output)

            # harvest ambiguities
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

            fixed, warnings = self._validate_and_fix(items, warnings, state)

            # --- Fallback detection and correction (batch) ---
            try:
                problem_segments = detect_missing_or_rejected_lines(
                    ch.text, fixed)
                print(
                    f"[DEBUG] Fallback detection: {len(problem_segments)} segment(s) found", flush=True)
                # [DIAG] validation: REJECTED present but no detected segments
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
                for _seg_i, seg in enumerate(problem_segments, start=1):
                    try:
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
                        fb_raw = call_frendli_fallback(
                            system_prompt, seg.get("text", ""), known_characters=list(state.known_characters or []))
                        print(
                            f"[DEBUG] Fallback raw returned length={len(fb_raw)}", flush=True)
                        try:
                            _fb_lines_preview = (fb_raw or "").splitlines()[:2]
                            print(
                                f"[DIAG] Fallback RAW (first 2 lines): {_fb_lines_preview}", flush=True)
                        except Exception:
                            pass
                        # optional: save raw friendli output when debug_save is enabled
                        if self.debug_save:
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
                        fb_lines = self._parse_jsonl(
                            self._preclean_jsonl(fb_raw))
                        print(
                            f"[DEBUG] Fallback parsed {len(fb_lines)} line(s)", flush=True)
                        fb_valid, warnings = self._validate_and_fix(
                            fb_lines, warnings, state)
                        print(
                            f"[DEBUG] Fallback validated {len(fb_valid)} line(s)", flush=True)
                        # [DIAG] unknown speakers produced by fallback
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
                        replace_or_insert_lines(fixed, seg.get("start_idx", 0), fb_valid, seg.get(
                            "end_idx", seg.get("start_idx", 0)))
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
                    for seg in problem_segments:
                        try:
                            fb_raw = call_frendli_fallback(
                                system_prompt,
                                seg.get("text", ""),
                                known_characters=list(
                                    state.known_characters or []),
                            )
                            fb_lines = self._parse_jsonl(
                                self._preclean_jsonl(fb_raw))
                            fb_valid, warnings = self._validate_and_fix(
                                fb_lines, warnings, state)
                            replace_or_insert_lines(
                                fixed,
                                seg.get("start_idx", 0),
                                fb_valid,
                                seg.get("end_idx", seg.get("start_idx", 0)),
                            )
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
            all_dialogues.extend(fixed)

        # Deduplicate overlapping outputs
        all_dialogues = deduplicate_lines(all_dialogues)

        # Reconcile adjacent same-speaker lines
        reconciled: List[Dict[str, Any]] = []
        for item in all_dialogues:
            if not self.include_narration and str(item.get("character")).strip() == "Narrator":
                continue
            if reconciled and reconciled[-1]["character"] == item["character"]:
                reconciled[-1]["text"] = f"{reconciled[-1]['text']} {item['text']}".strip()
                merged_emotions = list(dict.fromkeys(
                    reconciled[-1]["emotions"] + item["emotions"]))
                # enforce 2
                if len(merged_emotions) >= 2:
                    merged_emotions = merged_emotions[:2]
                else:
                    merged_emotions = ensure_two_emotions(
                        item["character"], merged_emotions, reconciled[-1]["text"], self.kb, self.allowed_emotions, self.memory)
                reconciled[-1]["emotions"] = merged_emotions
            else:
                reconciled.append(item)

        # --- Safeguard: re-inject missing input lines as Narrator (ambiguous) ---
        try:
            from .fallback_utils import _split_sentences as _split_sents, _normalize_text as _norm
            sent_infos: List[Dict[str, Any]] = _split_sents(raw_text)

            present = [_norm(d.get("text", "")) for d in reconciled]
            present_set = {p for p in present if p}

            import re as _re

            def _extract_quotes(s: str) -> List[str]:
                qs: List[str] = []
                try:
                    for pat in [r"\"([^\"]+)\"", r"“([^”]+)”,?", r"‘([^’]+)’", r"'([^']+)'"]:
                        for m in _re.findall(pat, s):
                            if m and m.strip():
                                qs.append(_norm(m))
                except Exception:
                    pass
                return qs

            missing_indices: List[int] = []
            for si in sent_infos:
                s_norm = si.get("norm", "")
                if not s_norm:
                    continue
                covered = (s_norm in present_set) or any(
                    (s_norm in p or p in s_norm) for p in present_set)
                if not covered and self.REINJECT_STRICT:
                    quotes = _extract_quotes(si.get("text", ""))
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
                if (not self.REINJECT_STRICT) or length >= 2:
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
                reconciled.append({
                    "character": "Narrator",
                    "emotions": ["neutral", "calm"],
                    "text": m,
                    "id": f"reinjected-{abs(hash(m))}",
                })
                try:
                    _samples = [d.get("text", "") for d in reconciled[:2]]
                    print(
                        f"[DIAG] Reinjected (strict) sentence: '{(m or '')[:120]}' | samples={_samples}", flush=True)
                except Exception:
                    pass
            _diag_reinj_added_total += len(to_reinject)
            if to_reinject:
                warnings.append(
                    f"Re-injected {len(to_reinject)} missing sentence(s) as Narrator.")
        except Exception as _se:
            # Never fail the main flow due to safeguard
            try:
                log_exception("parsers/openai_parser.py:convert", _se)
            except Exception:
                pass
            pass

        # Final purge of REJECTED lines before formatting
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

        stats = {
            "quotes_found": len(reconciled),
            "lines_emitted": len(formatted_lines),
            "narration_blocks": sum(1 for d in reconciled if d.get("character") == "Narrator"),
        }

        # [DIAG] End-of-run summary
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
            print("[DIAG] ===============================", flush=True)
        except Exception:
            pass

        try:
            log_to_session("INFO", "Parser convert end (success)",
                           src="parsers/openai_parser.py:convert")
        except Exception:
            pass
        return RawParseResult("\n".join(formatted_lines), reconciled, stats, ambiguities, warnings, errors)

    def _line_key(self, it: Dict[str, Any]) -> str:
        return hashlib.sha256(f"{(it.get('character') or '').strip().lower()}|{(it.get('text') or '').strip()}".encode("utf-8")).hexdigest()

    def convert_streaming(self, raw_text: str):
        warnings: List[str] = []
        errors: List[str] = []
        if not raw_text or not raw_text.strip():
            yield {"chunk_index": 0, "total_chunks": 0, "dialogues": [], "ambiguities": [], "warnings": warnings}
            return

        print(f">>> Building chunks (input chars={len(raw_text)})", flush=True)
        chunks = build_chunks(
            raw_text, max_tokens=self.max_tokens_per_chunk, model=self.model, overlap_sentences=2)
        print(f">>> Built {len(chunks)} chunks", flush=True)
        # Reset timings/tokens for this run
        self._run_call_durations = []
        self._run_chunk_token_counts = []
        state = ParserState(known_characters=set(
            self.default_known_characters))
        seen_keys: Set[str] = set()

        total = len(chunks)
        try:
            for idx, ch in enumerate(chunks):
                print(
                    f">>> Starting chunk {idx+1}/{total} (approx tokens={ch.token_count})", flush=True)
                summary = self._state_summary(state)
                system_prompt = build_system_prompt(self.allowed_emotions, list(
                    state.known_characters), self.include_narration, summary)
                user_prompt = build_user_prompt(ch.text, None)

                cache_key = _hash_key(ch.text, summary)
                chunk_duration_sec = 0.0
                if cache_key in self._cache:
                    print(">>> Using cached response", flush=True)
                    raw_output = self._cache[cache_key]
                    # cached response → duration 0.0
                    chunk_duration_sec = 0.0
                else:
                    try:
                        raw_output = self._call_openai(
                            system_prompt, user_prompt)
                        chunk_duration_sec += self._last_call_elapsed_sec
                    except Exception as e:
                        errors.append(
                            f"Parser API error on chunk {idx+1}/{total}: {e}")
                        # Surface partial progress then re-raise to allow fallback by caller
                        yield {"chunk_index": idx+1, "total_chunks": total, "dialogues": [], "ambiguities": [], "warnings": warnings, "errors": errors}
                        raise
                    if self.debug_save:
                        self._save_debug_output(
                            raw_output, suffix=f"_chunk{idx+1}")
                    self._cache[cache_key] = raw_output

                print(
                    f">>> Got response (chars={len(raw_output)})", flush=True)
                items = self._parse_jsonl(raw_output)
                print(f">>> Parsed {len(items)} items", flush=True)
                if not items:
                    print(">>> Retry with JSONL ONLY", flush=True)
                    raw_output = self._call_openai(
                        system_prompt + "\nIMPORTANT: Output JSONL ONLY.", user_prompt)
                    chunk_duration_sec += self._last_call_elapsed_sec
                    if self.debug_save:
                        self._save_debug_output(
                            raw_output, suffix=f"_retry_chunk{idx+1}")
                    items = self._parse_jsonl(raw_output)
                    print(
                        f">>> Parsed after retry: {len(items)} items", flush=True)

                # harvest ambiguities for this chunk
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

                fixed, warnings = self._validate_and_fix(
                    items, warnings, state)

                # --- Fallback detection and correction (streaming) ---
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
                    for _seg_i, seg in enumerate(problem_segments, start=1):
                        try:
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
                            fb_raw = call_frendli_fallback(
                                system_prompt, seg.get("text", ""), known_characters=list(state.known_characters or []))
                            print(
                                f"[DEBUG] Fallback raw returned length={len(fb_raw)}", flush=True)
                            try:
                                _fb_lines_preview = (
                                    fb_raw or "").splitlines()[:2]
                                print(
                                    f"[DIAG] Fallback RAW (first 2 lines): {_fb_lines_preview}", flush=True)
                            except Exception:
                                pass
                            if self.debug_save:
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
                            fb_lines = self._parse_jsonl(
                                self._preclean_jsonl(fb_raw))
                            print(
                                f"[DEBUG] Fallback parsed {len(fb_lines)} line(s)", flush=True)
                            fb_valid, warnings = self._validate_and_fix(
                                fb_lines, warnings, state)
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
                            replace_or_insert_lines(fixed, seg.get("start_idx", 0), fb_valid, seg.get(
                                "end_idx", seg.get("start_idx", 0)))
                        except Exception as _fe:
                            print(
                                f"[DEBUG] Fallback error ignored: {_fe}", flush=True)
                            continue
                else:
                    if any(d.get("character", "").upper() == "REJECTED" for d in fixed):
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
                                fb_lines = self._parse_jsonl(
                                    self._preclean_jsonl(fb_raw))
                                fb_valid, warnings = self._validate_and_fix(
                                    fb_lines, warnings, state)
                                replace_or_insert_lines(
                                    fixed,
                                    seg.get("start_idx", 0),
                                    fb_valid,
                                    seg.get("end_idx", seg.get(
                                        "start_idx", 0)),
                                )
                            except Exception as _fe:
                                print(
                                    f"[DIAG] Forced fallback failed (streaming): {_fe}", flush=True)
                                continue
                    else:
                        print(
                            "[DEBUG] No fallback needed for this chunk", flush=True)

                # incremental de-dup against seen_keys (due to overlap)
                filtered: List[Dict[str, Any]] = []
                for it in fixed:
                    key = self._line_key(it)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    filtered.append(it)

                # --- Streaming safeguard: re-inject missing sentences from this chunk as Narrator ---
                try:
                    from .fallback_utils import _split_sentences as _split_sents, _normalize_text as _norm
                    sent_infos = _split_sents(ch.text)
                    present = {_norm(d.get("text", "")) for d in filtered}
                    reinjected = 0
                    print(
                        f"[DIAG] Streaming reinjection: considered={len(sent_infos)}", flush=True)
                    import re as _re

                    def _extract_quotes(s: str) -> list[str]:
                        qs: list[str] = []
                        for pat in [r"\"([^\"]+)\"", r"“([^”]+)”,?", r"‘([^’]+)’", r"'([^']+)'"]:
                            for m in _re.findall(pat, s or ""):
                                if m and m.strip():
                                    qs.append(_norm(m))
                        return qs
                    missing_idx: list[int] = []
                    for si in sent_infos:
                        s_norm = si.get("norm", "")
                        if not s_norm:
                            continue
                        covered = (s_norm in present) or any(
                            (s_norm in p or p in s_norm) for p in present)
                        if not covered and self.REINJECT_STRICT:
                            qs = _extract_quotes(si.get("text", ""))
                            if qs and any(q in present for q in qs):
                                covered = True
                        if not covered:
                            missing_idx.append(si["index"])
                    # group and apply min gap >= 2 when strict
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
                        if (not self.REINJECT_STRICT) or length >= 2:
                            for si in sent_infos:
                                if a <= si["index"] <= b:
                                    filtered.append({
                                        "character": "Narrator",
                                        "emotions": ["neutral", "calm"],
                                        "text": si["text"],
                                        "id": f"reinjected-{abs(hash(si['text']))}",
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
                # Final purge of REJECTED lines before yielding (streaming)
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

                print(
                    f">>> Yielding chunk {idx+1}/{total}: dialogues={len(filtered)}, ambiguities={len(chunk_ambs)}", flush=True)
                # record per-chunk duration and token count
                self._run_call_durations.append(chunk_duration_sec)
                self._run_chunk_token_counts.append(
                    int(getattr(ch, "token_count", 0) or 0))
                yield {
                    "chunk_index": idx + 1,
                    "total_chunks": total,
                    "dialogues": filtered,
                    "ambiguities": chunk_ambs,
                    "warnings": warnings,
                }
        finally:
            # Append one CSV line with durations and token counts for this run
            try:
                log_path = Path(__file__).parent / "parser_logs.txt"
                entries: List[str] = []
                for i, d in enumerate(self._run_call_durations or []):
                    tok = 0
                    if i < len(self._run_chunk_token_counts):
                        tok = int(self._run_chunk_token_counts[i] or 0)
                    entries.append(f"{d:.2f} ({tok})")
                line = ", ".join(entries)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                print(f">>> Wrote parser timings/tokens: [{line}]", flush=True)
            except Exception as _e:
                print(f">>> Failed to write parser timings: {_e}", flush=True)

    def finalize_stream(self, dialogues: List[Dict[str, Any]], include_narration: Optional[bool] = None) -> RawParseResult:
        inc = self.include_narration if include_narration is None else include_narration
        # Preserve each dialogue line exactly as provided (no merging). Only apply narrator inclusion filter.
        reconciled: List[Dict[str, Any]] = []
        for item in dialogues:
            if not inc and str(item.get("character")).strip() == "Narrator":
                continue
            reconciled.append(item)

        formatted_lines: List[str] = []
        for d in reconciled:
            em_text = "".join([f"({e})" for e in d.get("emotions", [])])
            formatted_lines.append(
                f"[{d['character']}] {em_text}: {d['text']}".strip())

        stats = {
            "quotes_found": len(reconciled),
            "lines_emitted": len(formatted_lines),
            "narration_blocks": sum(1 for d in reconciled if d.get("character") == "Narrator"),
        }

        # collect ambiguities present in dialogues (those with id)
        ambiguities: List[Dict[str, Any]] = []
        for d in dialogues:
            if str(d.get("character", "")).lower() == "ambiguous":
                ambiguities.append({
                    "id": d.get("id") or f"amb-{abs(hash(d.get('text') or ''))}",
                    "text": d.get("text", ""),
                    "candidates": d.get("candidates", [])[:5] if isinstance(d.get("candidates"), list) else [],
                })

        return RawParseResult("\n".join(formatted_lines), reconciled, stats, ambiguities, [], [])
