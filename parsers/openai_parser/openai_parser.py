from __future__ import annotations
from difflib import SequenceMatcher as _SM
import re as _re_quotes
import re
from difflib import SequenceMatcher

import datetime
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import difflib
import time
import io
import sys

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field, ValidationError

from audio.utils import get_flat_emotion_tags, get_flat_character_voices
from settings import OPENAI_API_KEY
from .chunker import build_chunks, deduplicate_lines, deduplicate_lines_exact
from .chunker import diag_consume_dedup_conflicts
from .emotion_utils import EmotionMemory, build_emotion_kb, ensure_two_emotions, get_allowed_emotions
from .prompt_builder import build_system_prompt, build_user_prompt
from .fallback_utils import (
    detect_missing_or_rejected_lines,
    call_frendli_fallback,
    replace_or_insert_lines,
    filter_fallback_lines,
)
from utils.log_instrumentation import log_timed_action
from utils.session_logger import log_to_session, log_exception


class DualLogger(io.TextIOBase):
    """Duplicates stdout writes to both terminal and an in-memory buffer."""

    def __init__(self, original_stream):
        super().__init__()
        self.original = original_stream
        self.buffer = io.StringIO()

    def write(self, data):
        try:
            self.original.write(data)
            self.original.flush()
        except Exception:
            pass
        try:
            self.buffer.write(data)
        except Exception:
            pass
        return len(data)

    def flush(self):
        try:
            self.original.flush()
        except Exception:
            pass
        try:
            self.buffer.flush()
        except Exception:
            pass

    def get_value(self):
        try:
            return self.buffer.getvalue()
        except Exception:
            return ""


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

# Lightweight text similarity helper for EOD fallback filtering


def _token_similarity(a: str, b: str) -> float:
    """Lightweight text similarity check between two strings (0–1 scale)."""
    try:
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
    except Exception:
        return 0.0


def _fuzzy_sim(a: str, b: str) -> float:
    """SequenceMatcher-based similarity between two strings (0..1)."""
    try:
        a = (a or "").strip().lower()
        b = (b or "").strip().lower()
        return difflib.SequenceMatcher(None, a, b).ratio()
    except Exception:
        return 0.0


def _extract_quotes(text: str) -> str:
    """Extract spoken parts inside quotes for fair comparison against Friendli text."""
    try:
        quotes = _re_quotes.findall(r'“([^”]+)”|"([^"]+)"', text or "")
        if quotes:
            return " ".join([(q[0] or q[1]) for q in quotes])
    except Exception:
        pass
    return text or ""


def _dedupe_chunk_boundaries(all_chunks: List[List[Dict[str, Any]]], threshold: float = 0.9) -> List[Dict[str, Any]]:
    """
    Compare only the last line of each chunk with the first line of the next one.
    If similarity ≥ threshold, drop the earlier line (keep later chunk's version).
    Works safely with 1+ chunks.
    """
    if not all_chunks or len(all_chunks) <= 1:
        return [line for chunk in (all_chunks or []) for line in chunk]

    deduped: List[List[Dict[str, Any]]] = []
    for i, chunk in enumerate(all_chunks):
        if not chunk:
            continue
        if i == 0:
            deduped.append(chunk)
            continue

        prev_chunk = deduped[-1] if deduped else []
        current_chunk = chunk

        if not prev_chunk or not current_chunk:
            deduped.append(current_chunk)
            continue

        last_prev = prev_chunk[-1]
        first_curr = current_chunk[0]

        sim = _SM(None, (last_prev.get("text", "") or ""),
                  (first_curr.get("text", "") or "")).ratio()
        if sim >= threshold:
            try:
                print(
                    f"[DEDUP] Dropping overlapping line (sim={sim:.2f}): {((last_prev.get('text','') or '')[:80])!r}", flush=True)
            except Exception:
                pass
            prev_chunk = prev_chunk[:-1]

        if deduped:
            deduped[-1] = prev_chunk
        deduped.append(current_chunk)

    flat: List[Dict[str, Any]] = [line for chunk in deduped for line in chunk]
    try:
        print(
            f"[EOD][DEDUP_SUMMARY] merged_chunks={len(all_chunks)} → after_dedup={len(flat)} lines", flush=True)
    except Exception:
        pass
    return flat


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
        self._eod_fallback_throttled: bool = False
        self.legacy_base_parser: bool = True  # TODO: later wire via settings if needed

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

            # Instrumentation: narrator/ambiguous attribution
            try:
                if (fixed.get("character") in ("Narrator", "Ambiguous")):
                    print(
                        f"[ATTR] {fixed['character']} text='{fixed['text'][:80]}'",
                        flush=True,
                    )
            except Exception:
                pass

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
        if not objs:
            try:
                print(
                    "[WARN] _parse_jsonl: no valid JSONL lines extracted", flush=True)
            except Exception:
                pass
        return objs

    def _normalize_text_for_match(self, s: str) -> str:
        if not s:
            return ""
        # unify quotes and whitespace; lowercase
        s = s.replace("\u201C", '"').replace("\u201D", '"')
        s = s.replace("\u2018", "'").replace("\u2019", "'")
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    def _build_sentence_to_pos_map(self, chunk_text: str, dialogues: list) -> dict:
        """
        Map sentence index → earliest dialogue index that best covers that sentence.
        We consider containment or high similarity against dialogue text.
        """
        # Robust split: similar to our reinjection splitter
        def _split_sentences_robust(text: str) -> list:
            t = (text or "")
            t = t.replace("\u201C", '"').replace("\u201D", '"')
            t = t.replace("\u2018", "'").replace("\u2019", "'")
            t = t.replace("\u2026", "...")
            boundary = re.compile(
                r'(?<=[.!?])\s+'
                r'|(?<=,")\s+'
                r'|(?<=,”)\s+'
                r'|(?<=")\s+(?=(?:[A-Z]|he|she|they))'
                r'|(?<=\.)\s+(?=")'
            )
            parts = [p.strip() for p in boundary.split(t) if p and p.strip()]
            return parts

        sents = _split_sentences_robust(chunk_text)
        sent_norms = [self._normalize_text_for_match(s) for s in sents]

        pos_map = {}  # sent_idx -> dialogue_pos
        for di, obj in enumerate(dialogues):
            txt = self._normalize_text_for_match((obj or {}).get("text", ""))
            if not txt:
                continue
            # find the best sentence idx that this dialogue most likely covers
            best_idx, best_sim = None, 0.0
            for si, sn in enumerate(sent_norms):
                if not sn:
                    continue
                if sn and (sn in txt or txt in sn):
                    # containment is strong signal; prefer first hit
                    best_idx, best_sim = si, 1.0
                    break
                sim = SequenceMatcher(None, sn, txt).ratio()
                if sim > best_sim:
                    best_idx, best_sim = si, sim
            if best_idx is not None:
                pos_map.setdefault(best_idx, di)
        return pos_map

    def _simple_reinject_missing_as_narrator(self, original_text: str, lines: list) -> list:
        """
        Legacy behavior: ensure every input sentence is represented.
        - Split original_text into simple sentences.
        - Normalize and compare against produced lines' text.
        - For any sentence not covered by any line, append a Narrator line with that exact sentence.
        - Preserve input order for any injected lines by scanning in input sequence and appending in that order.
        """
        import re
        from difflib import SequenceMatcher

        def _split_sentences_robust(text: str) -> list[str]:
            """
            Robust sentence splitter for reinjection:
            - Treats standard sentence endings (. ! ?) as boundaries.
            - Also treats comma+closing-quote (,” or ,") as a boundary commonly found in dialogue:
              e.g., “You found it,” he said.
            - Handles curly quotes (“ ”), ellipsis (…) and em dashes (—).
            - Avoids over-splitting on simple commas.
            """
            import re

            if not text:
                return []

            # Normalize common punctuation variants to make regex matching reliable
            t = text
            t = t.replace("\u201C", '"').replace(
                "\u201D", '"')   # curly double quotes → "
            t = t.replace("\u2018", "'").replace(
                "\u2019", "'")   # curly single quotes → '
            t = t.replace("\u2026", "...")                        # ellipsis
            # Don't remove em dash — leave it; not a hard boundary by itself

            # Pattern explanation:
            #  - (?<=[.!?])\s+        → standard sentence enders
            #  - (?<=,")\s+           → comma + closing quote boundary
            #  - (?<=,”)\s+           → comma + curly quote boundary (already normalized above, but keep for safety)
            #  - (?<=")\s+(?=(?:[A-Z]|he|she|they)) → closing quote followed by a likely attribution/next clause
            #  - (?<=\.)\s+(?=")      → period followed by opening quote (new sentence starting with a quote)
            #
            # Notes:
            #  * We deliberately do NOT split on plain commas.
            #  * The attribution lookahead (he|she|they|Capitalized) is a pragmatic heuristic for dialogue attribution tails.
            #  * Order of alternatives is chosen to prefer clear hard boundaries first.
            boundary = re.compile(
                r'(?<=[.!?])\s+'
                r'|(?<=,")\s+'
                r'|(?<=,”)\s+'
                r'|(?<=")\s+(?=(?:[A-Z]|he|she|they))'
                r'|(?<=\.)\s+(?=")'
            )

            # Split and trim; keep only non-empty sentences
            parts = [p.strip() for p in boundary.split(t) if p and p.strip()]
            return parts

        def _normalize(s: str) -> str:
            return re.sub(r"\s+", " ", s).strip().lower()

        # robust sentence split to handle dialogue clauses ending with comma+quote
        raw_sents = _split_sentences_robust(original_text)
        raw_sents = [s for s in raw_sents if s.strip()]

        produced_texts_norm = []
        for obj in lines:
            try:
                txt = (obj.get("text", "") or "")
                txt = txt.replace("\u201C", '"').replace(
                    "\u201D", '"').replace("\u2018", "'").replace("\u2019", "'")
                produced_texts_norm.append(_normalize(txt))
            except Exception:
                produced_texts_norm.append("")

        def _covered(sent_norm: str) -> bool:
            # Containment OR high-similarity match is considered covered
            for pt in produced_texts_norm:
                if not pt:
                    continue
                if sent_norm in pt or pt in sent_norm:
                    return True
                if SequenceMatcher(None, sent_norm, pt).ratio() >= 0.92:
                    return True
            return False

        reinjected = []
        for s in raw_sents:
            sn = _normalize(s)
            if not _covered(sn):
                reinjected.append({
                    "character": "Narrator",
                    "emotions": ["neutral", "neutral"],
                    "text": s,
                    "_src": "reinj"
                })
        if not reinjected:
            return lines

        # Append reinjected lines at the end (legacy simple approach)
        return lines + reinjected

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
        per_chunk_dialogues: List[List[Dict[str, Any]]] = []
        ambiguities: List[Dict[str, Any]] = []

        for idx, ch in enumerate(chunks):
            if idx == len(chunks) - 1:
                print(
                    "\n===== [EOD][CHUNK_TEXT] LAST CHUNK INPUT BEGIN =====", flush=True)
                try:
                    print(ch.text, flush=True)
                except Exception:
                    print("[EOD] <failed to print chunk text>", flush=True)
                print(
                    "===== [EOD][CHUNK_TEXT] LAST CHUNK INPUT END =====\n", flush=True)
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
            # Tag source for DIAG
            for _ln in fixed:
                try:
                    _ln.setdefault("_src", "ai")
                except Exception:
                    pass

            # --- Fallback detection and correction (batch) ---
            try:
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
                # --- EOD fallback safety filter ---
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
                        from .fallback_utils import _split_sentences as _fb_split
                        # Merge text but preserve approximate indices for span mapping
                        combined_text = " ".join(
                            seg.get("text", "") for seg in problem_segments[:15])
                        # Approximate indices flattened from original groups
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
                        # Ensure at least one index exists
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
                        self._eod_fallback_throttled = True
                    else:
                        self._eod_fallback_throttled = False
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
                        # Scope known characters to current chunk's active speakers (+Narrator)
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
                        # If EOD consolidation carried approx indices, assign them to lines by order
                        try:
                            approx = seg.get("_approx_indices") or []
                            if approx and fb_valid:
                                for k, _ln in enumerate(fb_valid):
                                    _ln["_span_start"] = int(
                                        approx[min(k, len(approx)-1)])
                        except Exception:
                            pass
                        # --- Localized fuzzy validation against segment text + intra-response dedup ---
                        segment_text = seg.get("text", "") or ""
                        from .fallback_utils import filter_fallback_lines as _fb_filter
                        # Apply robust fuzzy filter with diagnostics
                        fb_valid = _fb_filter(segment_text, fb_valid)
                        print(
                            f"[EOD][FB_SUMMARY] After fuzzy filter: kept={len(fb_valid)}", flush=True)
                        for _ln in fb_valid:
                            try:
                                _ln.setdefault("_src", "fb")
                            except Exception:
                                pass
                        # Attach deterministic insertion position based on sentence span mapping
                        try:
                            sent_to_pos = self._build_sentence_to_pos_map(
                                ch.text, fixed)
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
                        if idx == len(chunks) - 1:
                            print(
                                f"[EOD][FB] validated_and_kept={len(fb_valid)} line(s) for seg#{_seg_i}", flush=True)
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
                        replace_or_insert_lines(fixed, seg.get("start_idx", 0), fb_valid, seg.get(
                            "end_idx", seg.get("start_idx", 0)))
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
                    for seg in problem_segments:
                        try:
                            # Scope known characters to current chunk's active speakers (+Narrator)
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
                            fb_lines = self._parse_jsonl(
                                self._preclean_jsonl(fb_raw))
                            fb_valid, warnings = self._validate_and_fix(
                                fb_lines, warnings, state)
                            # Propagate span start for positional reinsertion (streaming)
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
            # store per-chunk parsed lines (post-fallback and validation)
            per_chunk_dialogues.append(list(fixed))
            all_dialogues.extend(fixed)

        # Deduplicate overlapping outputs
        # Boundary de-duplication across chunk edges, then standard dedup
        try:
            merged_parsed_lines = _dedupe_chunk_boundaries(per_chunk_dialogues)
        except Exception:
            merged_parsed_lines = list(all_dialogues)
        if self.legacy_base_parser:
            # Apply conservative exact dedup for legacy mode
            all_dialogues = deduplicate_lines_exact(merged_parsed_lines)
        else:
            # Apply newer dedup behavior
            all_dialogues = deduplicate_lines(merged_parsed_lines)

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
            full_text = raw_text
            if self.legacy_base_parser:
                # Use simple narrator reinjection that guarantees 100% sentence coverage
                reconciled = self._simple_reinject_missing_as_narrator(
                    full_text, reconciled)
            else:
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
                    # Allow single short narrative reinjection if under 12 tokens
                    sent_texts = [si["text"]
                                  for si in sent_infos if a <= si["index"] <= b]
                    avg_tokens = sum(len(t.split())
                                     for t in sent_texts) / max(1, len(sent_texts))
                # Instrumentation: reinjection group summary
                try:
                    print(
                        f"[REINJ_GROUP] span=({a},{b}) count={len(sent_texts)}", flush=True)
                    for _txt in sent_texts:
                        print(
                            f"    [REINJ_LINE] text='{(_txt or '')[:80]}'", flush=True)
                except Exception:
                    pass
                    if (not self.REINJECT_STRICT) or length >= 2 or avg_tokens <= 12:
                        if self.REINJECT_STRICT and length < 2 and avg_tokens <= 12:
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
                    # infer character contextually from last speaker; default Narrator
                    try:
                        last_char = reconciled[-1]["character"] if reconciled else "Narrator"
                    except Exception:
                        last_char = "Narrator"
                    txt_lower = (m or "").lower()
                    char_guess = last_char
                    try:
                        if any(w in txt_lower for w in ["“", '"', " said", "asked", "whispered", "replied", "murmured", "moaned", "commanded"]) and last_char not in ("Narrator", "Ambiguous"):
                            char_guess = last_char
                        elif (m or "").strip().startswith(("“", '"')):
                            # quote but no clear attribution
                            char_guess = "Ambiguous" if last_char in (
                                "Narrator", "Ambiguous") else last_char
                        else:
                            char_guess = "Narrator"
                    except Exception:
                        char_guess = last_char or "Narrator"
                    reconciled.append({
                        "character": char_guess,
                        "emotions": ["neutral", "calm"],
                        "text": m,
                        "id": f"reinjected-{abs(hash(m))}",
                        "_src": "reinj",
                    })
                    try:
                        reason = "prev" if char_guess not in ("Narrator", "Ambiguous") else (
                            "quote-no-speaker" if (m or "").strip().startswith(("“", '"')) else "none")
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
            # Never fail the main flow due to safeguard
            try:
                log_exception("parsers/openai_parser.py:convert", _se)
            except Exception:
                pass
            pass

        # Print final reconciled tail for last 20 lines (pre-REJECTED-purge)
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
                print("===== [EOD][TAIL] END =====\n", flush=True)
            except Exception:
                print("[EOD] <failed to print reconciled tail>", flush=True)

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

        # Instrumentation: ordering view on last 20 reconciled lines
        try:
            for i, d in enumerate(reconciled[-20:]):
                print(
                    f"[ORDER] idx={i:04d} src={d.get('_src')} char={d.get('character')} text='{(d.get('text') or '')[:60]}'",
                    flush=True,
                )
        except Exception:
            pass

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
        # --- End unified debug capture ---
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

    def _line_key(self, it: Dict[str, Any]) -> str:
        return hashlib.sha256(f"{(it.get('character') or '').strip().lower()}|{(it.get('text') or '').strip()}".encode("utf-8")).hexdigest()

    def convert_streaming(self, raw_text: str):
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
                # Tag source for DIAG
                for _ln in fixed:
                    try:
                        _ln.setdefault("_src", "ai")
                    except Exception:
                        pass

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
                    # --- EOD fallback safety filter (streaming) ---
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
                            self._eod_fallback_throttled = True
                        else:
                            self._eod_fallback_throttled = False
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
                            # Scope known characters to current chunk's active speakers (+Narrator)
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
                            # If EOD consolidation carried approx indices, assign them to lines by order
                            try:
                                approx = seg.get("_approx_indices") or []
                                if approx and fb_valid:
                                    for k, _ln in enumerate(fb_valid):
                                        _ln["_span_start"] = int(
                                            approx[min(k, len(approx)-1)])
                            except Exception:
                                pass
                            # --- Localized fuzzy validation against segment text + intra-response dedup (streaming) ---
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
                            replace_or_insert_lines(fixed, seg.get("start_idx", 0), fb_valid, seg.get(
                                "end_idx", seg.get("start_idx", 0)))
                            if is_last_chunk:
                                print(
                                    f"[EOD][FB_SUMMARY] After filtering: kept={len(fb_valid)} lines for last chunk.", flush=True)
                        except Exception as _fe:
                            print(
                                f"[DEBUG] Fallback error ignored: {_fe}", flush=True)
                            continue
                else:
                    if any(d.get("character", "").upper() == "REJECTED" for d in fixed):
                        if getattr(self, "_eod_fallback_throttled", False):
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
                    if self.legacy_base_parser:
                        filtered = self._simple_reinject_missing_as_narrator(
                            ch.text, filtered)
                    else:
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
                            # Allow single short narrative reinjection if under 12 tokens
                            sent_texts = [si["text"]
                                          for si in sent_infos if a <= si["index"] <= b]
                            avg_tokens = sum(len(t.split())
                                             for t in sent_texts) / max(1, len(sent_texts))
                            if (not self.REINJECT_STRICT) or length >= 2 or avg_tokens <= 12:
                                if self.REINJECT_STRICT and length < 2 and avg_tokens <= 12:
                                    try:
                                        print(
                                            f"[DIAG][REINJECT] (streaming) single-line reinjection enabled (avg_tokens={avg_tokens:.1f}) span=[{a}:{b}]", flush=True)
                                    except Exception:
                                        pass
                                for si in sent_infos:
                                    if a <= si["index"] <= b:
                                        filtered.append({
                                            "character": "Narrator",
                                            "emotions": ["neutral", "calm"],
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
                # record per-chunk duration and token count
                self._run_call_durations.append(chunk_duration_sec)
                self._run_chunk_token_counts.append(
                    int(getattr(ch, "token_count", 0) or 0))
                # Instrumentation: warn when no lines will be yielded for this chunk
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
            # --- End unified debug capture ---
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
