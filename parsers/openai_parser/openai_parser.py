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
from .emotion_utils import EmotionMemory, build_emotion_kb, ensure_two_emotions, get_allowed_emotions
from .prompt_builder import build_system_prompt, build_user_prompt


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
    def __init__(
        self,
        model: str = "gpt-5-mini",
        include_narration: bool = True,
        max_tokens_per_chunk: int = 1000,
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

    def convert(self, raw_text: str) -> RawParseResult:
        warnings: List[str] = []
        errors: List[str] = []
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
            raw_lines: List[str] = [ln.strip()
                                    for ln in (raw_text or "").split("\n")]
            raw_lines = [ln for ln in raw_lines if ln]

            def _norm(s: str) -> str:
                return (s or "").strip().lower()

            out_texts_lower: List[str] = [
                _norm(d.get("text", "")) for d in reconciled]
            missing_lines: List[str] = []
            for ln in raw_lines:
                ln_norm = _norm(ln)
                if not any(ln_norm in t for t in out_texts_lower):
                    missing_lines.append(ln)

            for m in missing_lines:
                reconciled.append({
                    "character": "Narrator",
                    "emotions": ["neutral", "calm"],
                    "text": m,
                    "id": f"reinjected-{abs(hash(m))}",
                })
            if missing_lines:
                warnings.append(
                    f"Re-injected {len(missing_lines)} missing line(s) as Narrator.")
        except Exception:
            # Never fail the main flow due to safeguard
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
                    import re as _re
                    exp_sentences: List[str] = []
                    for sent in _re.split(r"(?<=[.!?])\s+", (ch.text or "").strip()):
                        s = (sent or "").strip()
                        if s:
                            exp_sentences.append(s)

                    def _norm(s: str) -> str:
                        return (s or "").strip().lower()
                    present = {_norm(d.get("text", "")) for d in filtered}
                    reinjected = 0
                    for s in exp_sentences:
                        if _norm(s) not in present:
                            filtered.append({
                                "character": "Narrator",
                                "emotions": ["neutral", "calm"],
                                "text": s,
                                "id": f"reinjected-{abs(hash(s))}",
                            })
                            reinjected += 1
                    if reinjected:
                        warnings.append(
                            f"Streaming: re-injected {reinjected} missing sentence(s) as Narrator.")
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
        # Reconcile adjacent same-speaker lines
        reconciled: List[Dict[str, Any]] = []
        for item in dialogues:
            if not inc and str(item.get("character")).strip() == "Narrator":
                continue
            if reconciled and reconciled[-1]["character"] == item["character"]:
                reconciled[-1]["text"] = f"{reconciled[-1]['text']} {item['text']}".strip()
                merged_emotions = list(dict.fromkeys(
                    reconciled[-1]["emotions"] + item["emotions"]))
                if len(merged_emotions) >= 2:
                    merged_emotions = merged_emotions[:2]
                else:
                    merged_emotions = ensure_two_emotions(
                        item["character"], merged_emotions, reconciled[-1]["text"], self.kb, self.allowed_emotions, self.memory)
                reconciled[-1]["emotions"] = merged_emotions
            else:
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
